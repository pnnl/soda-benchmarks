// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
`timescale 1ns/1ps
// Validation integration: retain the existing ESP socket's module/port names.
// forward_kernel and panda_libtech are compiled directly from the supplied tar.
// Read requests use 32-bit word indices/lengths; ESP uses 64-bit beat indices.
// Reads may start/end on either half. Writes must cover aligned whole beats
// (true for this GEMM); no read-modify-write or partial-write support is claimed.
// Stage complete write chunks before occupying ESP's single DMA engine: Bambu
// can publish a write command before the reads needed to produce its data.
module sqr_bambu_basic_dma64 #(
`ifdef SODAP_LCS_GENERIC
  parameter CHECK_ARCHIVE_GEMM = 0,
  parameter TIMEOUT_CYCLES = 1000000,
`else
  parameter CHECK_ARCHIVE_GEMM = 1,
  parameter TIMEOUT_CYCLES = 100000,
`endif
  parameter WRITE_BUFFER_BEATS = 16
) (
  input clk, rst, conf_done,
  output reg acc_done,
  output [31:0] debug,
  output dma_read_ctrl_valid,
  output [31:0] dma_read_ctrl_data_index, dma_read_ctrl_data_length,
  output [2:0] dma_read_ctrl_data_size,
  output [5:0] dma_read_ctrl_data_user,
  input dma_read_ctrl_ready,
  input dma_read_chnl_valid,
  input [63:0] dma_read_chnl_data,
  output dma_read_chnl_ready,
  output dma_write_ctrl_valid,
  output [31:0] dma_write_ctrl_data_index, dma_write_ctrl_data_length,
  output [2:0] dma_write_ctrl_data_size,
  output [5:0] dma_write_ctrl_data_user,
  input dma_write_ctrl_ready,
  output dma_write_chnl_valid,
  output [63:0] dma_write_chnl_data,
  input dma_write_chnl_ready
);
  wire [63:0] read_request, write_request;
  wire read_request_valid, write_request_valid, read_request_ready, write_request_ready;
  wire core_done, core_read_ready, core_write_valid, core_write_ready;
  wire [31:0] core_write_data;
  reg read_active, read_full, read_half;
  reg [31:0] read_words_left;
  reg [63:0] read_buffer;
  reg write_half, done_pending;
  localparam W_IDLE=2'd0, W_FILL=2'd1, W_REQUEST=2'd2, W_SEND=2'd3;
  localparam WCOUNT_BITS = $clog2(WRITE_BUFFER_BEATS+1);
  reg [1:0] write_state;
  reg [31:0] write_beats_left, write_low, write_index, write_chunk_length;
  reg [WCOUNT_BITS-1:0] write_fill_count, write_send_index;
  reg [63:0] write_buffer [0:WRITE_BUFFER_BEATS-1];
  wire write_active = write_state != W_IDLE;
  wire write_full = write_state == W_REQUEST || write_state == W_SEND;
  reg core_start;

  // conf_done is asserted in ESP's CONFIG state. The core can produce its
  // first request combinationally from start. Delay it until ESP is RUNNING:
  // otherwise the independent TLB accepts the request during CONFIG, and
  // pending_dma_read makes the DMA FSM skip the initial request/grant phase.
  always @(posedge clk) begin
    if (!rst) core_start <= 0;
    else core_start <= conf_done;
  end

  // A bounded simulation-only request stall exercises ping-pong ownership.
  integer stall_until = 0;
  integer cycles = 0;
  // synthesis translate_off
  initial if (!$value$plusargs("gemm_stall=%d", stall_until)) stall_until = 0;
  // synthesis translate_on
  wire allow_write_request = cycles >= stall_until;
  always @(posedge clk) begin
    if (!rst || conf_done) cycles <= 0;
    else if (cycles < 100000000) cycles <= cycles + 1;
  end

  forward_kernel u_core (
    .clock(clk), .reset(rst), .start_port(core_start), .done_port(core_done), .idle_port(),
    .P0_TDATA(read_request), .P0_TVALID(read_request_valid), .P0_TREADY(read_request_ready),
    .P1_TDATA(read_half ? read_buffer[63:32] : read_buffer[31:0]),
    .P1_TVALID(read_full), .P1_TREADY(core_read_ready),
    .P2_TDATA(write_request), .P2_TVALID(write_request_valid), .P2_TREADY(write_request_ready),
    .P3_TDATA(core_write_data), .P3_TVALID(core_write_valid), .P3_TREADY(core_write_ready)
  );

  // ceil((first_lane + requested_words)/2); retain a carry for large lengths.
  wire [32:0] rounded_read_words = {1'b0,read_request[63:32]} +
                                  {32'b0,read_request[0]} + 33'd1;
  assign dma_read_ctrl_valid = read_request_valid && !read_active;
  assign read_request_ready = dma_read_ctrl_ready && !read_active;
  assign dma_read_ctrl_data_index = read_request[31:0] >> 1;
  assign dma_read_ctrl_data_length = rounded_read_words[32:1];
  assign dma_read_ctrl_data_size = 3'd3;
  assign dma_read_ctrl_data_user = 0;
  assign dma_read_chnl_ready = read_active && !read_full;

  assign dma_write_ctrl_valid = write_state == W_REQUEST && allow_write_request;
  assign write_request_ready = write_state == W_IDLE;
  assign dma_write_ctrl_data_index = write_index;
  assign dma_write_ctrl_data_length = write_chunk_length;
  assign dma_write_ctrl_data_size = 3'd3;
  assign dma_write_ctrl_data_user = 0;
  assign core_write_ready = write_state == W_FILL;
  assign dma_write_chnl_valid = write_state == W_SEND;
  assign dma_write_chnl_data = write_buffer[write_send_index];
  assign debug = {26'b0, done_pending, write_active, write_full, write_half, read_active, read_full};

  always @(posedge clk) begin
    if (!rst) begin
      read_active <= 0; read_full <= 0; read_half <= 0;
      read_words_left <= 0; read_buffer <= 0;
      write_state <= W_IDLE; write_half <= 0;
      write_beats_left <= 0; write_low <= 0; write_index <= 0;
      write_chunk_length <= 0; write_fill_count <= 0; write_send_index <= 0;
      done_pending <= 0; acc_done <= 0;
    end else begin
      acc_done <= 0;
      if (dma_read_ctrl_valid && dma_read_ctrl_ready) begin
        read_active <= 1;
        read_words_left <= read_request[63:32];
        read_half <= read_request[0];
      end
      if (dma_read_chnl_valid && dma_read_chnl_ready) begin
        read_buffer <= dma_read_chnl_data;
        read_full <= 1;
      end
      if (read_full && core_read_ready) begin
        read_words_left <= read_words_left - 1;
        if (read_words_left == 1) begin
          // Discard the unused high half when a request ends on the low half.
          read_active <= 0; read_full <= 0; read_half <= 0;
        end else if (read_half) begin
          read_full <= 0; read_half <= 0;
        end else read_half <= 1;
      end
      // Consume the core's descriptor independently of ESP. Its output data
      // may depend on further reads, so never reserve ESP's DMA engine yet.
      if (write_request_valid && write_request_ready) begin
        write_state <= W_FILL;
        write_index <= write_request[31:0] >> 1;
        write_beats_left <= write_request[63:32] >> 1;
        write_chunk_length <= (write_request[63:32] >> 1) > WRITE_BUFFER_BEATS ?
                              WRITE_BUFFER_BEATS : (write_request[63:32] >> 1);
        write_fill_count <= 0; write_send_index <= 0; write_half <= 0;
      end
      if (core_write_valid && core_write_ready) begin
        if (!write_half) begin write_low <= core_write_data; write_half <= 1; end
        else begin
          write_buffer[write_fill_count] <= {core_write_data, write_low};
          write_fill_count <= write_fill_count + 1;
          write_half <= 0;
          if (write_fill_count + 1 == write_chunk_length)
            write_state <= W_REQUEST;
        end
      end
      if (dma_write_ctrl_valid && dma_write_ctrl_ready) begin
        write_state <= W_SEND;
        write_send_index <= 0;
      end
      if (dma_write_chnl_valid && dma_write_chnl_ready) begin
        if (write_send_index + 1 == write_chunk_length) begin
          write_beats_left <= write_beats_left - write_chunk_length;
          write_index <= write_index + write_chunk_length;
          write_fill_count <= 0; write_send_index <= 0;
          if (write_beats_left == write_chunk_length)
            write_state <= W_IDLE;
          else begin
            write_state <= W_FILL;
            write_chunk_length <= (write_beats_left - write_chunk_length) > WRITE_BUFFER_BEATS ?
                                  WRITE_BUFFER_BEATS : (write_beats_left - write_chunk_length);
          end
        end else write_send_index <= write_send_index + 1;
      end
      if (core_done) done_pending <= 1;
      if (done_pending && !write_active && !write_full && !write_half && !read_active) begin
        acc_done <= 1; done_pending <= 0;
      end
    end
  end

  // Passive simulation checks. No operand data is supplied by this monitor.
  // synthesis translate_off
  integer invocation = 0, nr, nw, read_words, read_beats, write_words, write_beats, blocked;
  integer simulation_timeout = TIMEOUT_CYCLES;
  initial begin
    if ($value$plusargs("gemm_timeout=%d", simulation_timeout)) begin
      if (simulation_timeout < 1 || simulation_timeout >= 100000000)
        $fatal(1,"gemm_timeout must be between 1 and 99999999 cycles");
    end
  end
  initial if (WRITE_BUFFER_BEATS < 1) $fatal(1,"WRITE_BUFFER_BEATS must be positive");
  reg monitoring_run = 0;
  reg rr_held = 0, wr_held = 0, wd_held = 0;
  reg [63:0] rr_previous, wr_previous, wd_previous;
  always @(posedge clk) begin
    if (conf_done) begin
      monitoring_run = 1;
      invocation = invocation + 1;
      nr=0; nw=0; read_words=0; read_beats=0; write_words=0; write_beats=0; blocked=0;
      $display("GEMM_BOUNDARY START invocation=%0d stall_until=%0d", invocation, stall_until);
    end
    if (rst && invocation > 0) begin
      if (monitoring_run && (cycles == 16 || cycles == 1000))
        $display("GEMM_HANDSHAKE cycles=%0d rst=%b core_req=%b req=%h socket_req=%b grant=%b active=%b core_done=%b",
                 cycles,rst,read_request_valid,read_request,dma_read_ctrl_valid,dma_read_ctrl_ready,read_active,core_done);
      if (monitoring_run && cycles == simulation_timeout) $fatal(1,"GEMM hardware timeout");
      if (rr_held && (!dma_read_ctrl_valid ||
          {dma_read_ctrl_data_length,dma_read_ctrl_data_index} !== rr_previous))
        $fatal(1,"GEMM read request changed while stalled");
      if (wr_held && (!dma_write_ctrl_valid ||
          {dma_write_ctrl_data_length,dma_write_ctrl_data_index} !== wr_previous))
        $fatal(1,"GEMM write request changed while stalled");
      if (wd_held && (!dma_write_chnl_valid || dma_write_chnl_data !== wd_previous))
        $fatal(1,"GEMM write data changed while stalled");
      rr_held = dma_read_ctrl_valid && !dma_read_ctrl_ready;
      wr_held = dma_write_ctrl_valid && !dma_write_ctrl_ready;
      wd_held = dma_write_chnl_valid && !dma_write_chnl_ready;
      rr_previous = {dma_read_ctrl_data_length,dma_read_ctrl_data_index};
      wr_previous = {dma_write_ctrl_data_length,dma_write_ctrl_data_index};
      wd_previous = dma_write_chnl_data;
      if (write_request_valid && !write_request_ready) blocked++;
      if (dma_read_ctrl_valid && dma_read_ctrl_ready) begin
        if (read_request[63:32] == 0 ||
            (CHECK_ARCHIVE_GEMM && read_request[31:0]+read_request[63:32] > 514))
          $fatal(1,"GEMM invalid read request %h", read_request);
        if (CHECK_ARCHIVE_GEMM && nr < 2) begin
          if (read_request !== {32'd1, 32'(512+nr)} || dma_read_ctrl_data_length != 1 ||
              dma_read_ctrl_data_index != 256) $fatal(1,"GEMM scalar adapter failure");
          $display("GEMM_SCALAR word_index=%0d words=1 beat_index=%0d beats=%0d lane=%0d",
                   read_request[31:0], dma_read_ctrl_data_index, dma_read_ctrl_data_length, read_request[0]);
        end
        nr++;
      end
      if (read_full && core_read_ready) read_words++;
      if (dma_read_chnl_valid && dma_read_chnl_ready) read_beats++;
      if (write_request_valid && write_request_ready) begin
        if (write_request[0] || write_request[32] || write_request[63:32] == 0 ||
            (CHECK_ARCHIVE_GEMM && (write_request[63:32] != 4 ||
             write_request[31:0] < 416 || write_request[31:0]+write_request[63:32] > 512)))
          $fatal(1,"GEMM unsupported write request %h", write_request);
      end
      if (dma_write_ctrl_valid && dma_write_ctrl_ready) nw++;
      if (core_write_valid && core_write_ready) write_words++;
      if (dma_write_chnl_valid && dma_write_chnl_ready) write_beats++;
      if (acc_done) begin
        monitoring_run = 0;
        $display("GEMM_BOUNDARY RESULT invocation=%0d requests=%0d/%0d words=%0d/%0d beats=%0d/%0d cycles=%0d blocked=%0d",
                 invocation,nr,nw,read_words,write_words,read_beats,write_beats,cycles,blocked);
        if (CHECK_ARCHIVE_GEMM && (nr!=218 || nw!=24 || read_words!=866 || write_words!=96 || read_beats!=434 || write_beats!=48))
          $fatal(1,"GEMM DMA count mismatch");
      end
    end
  end
  // synthesis translate_on
endmodule
