// Fixture: the ESP DMA ping-pong accelerator (load -> compute -> store) in the
// call-graph form sodap-dataflow-nodes-to-func produces. The four streams of
// @top carry the DMA read/write requests and data; ctrl1/ctrl2 and the two
// buffers are internal.

module {
  // ------------------------------------------------------------
  // Load node
  //
  // For each batch:
  //   1. Construct DMA read request:
  //        (16 << 32) | base_index
  //   2. Send request to DMA
  //   3. Read 16 values from DMA
  //   4. Store them into the selected ping-pong buffer
  //   5. Send the buffer index to compute
  // ------------------------------------------------------------

  func.func @top_node0(
      %arg0 : !dataflow.stream<i64, 8>,    // dma_read_ctrl
      %arg1 : !dataflow.stream<i32, 16>,   // dma_read_data
      %arg2 : memref<2x16xi32>,            // buffer1
      %arg3 : !dataflow.stream<i8, 2>) {   // ctrl1

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index

    %c32_i64 = arith.constant 32 : i64
    %c16_i64 = arith.constant 16 : i64

    affine.for %b = 0 to 8 {

      // ----------------------------------------------------------
      // base_index = b * BATCH_ITEM_NUMBER
      // ----------------------------------------------------------
      %base = affine.apply affine_map<(d0) -> (d0 * 16)>(%b)

      // ----------------------------------------------------------
      // Pack DMA request:
      //
      //     request = (16 << 32) | base_index
      //
      // This corresponds to:
      //     dma_req_t req =
      //         ((uint64_t)16 << 32) | base_index;
      // ----------------------------------------------------------
      %base_i64 = arith.index_cast %base : index to i64
      %len_shifted = arith.shli %c16_i64, %c32_i64 : i64
      %req = arith.ori %len_shifted, %base_i64 : i64

      dataflow.stream_write %arg0, %req
          : <i64, 8>, i64

      // ----------------------------------------------------------
      // Ping-pong buffer index:
      //
      //     buffer_index = b % 2
      // ----------------------------------------------------------
      %buffer_index = affine.apply
          affine_map<(d0) -> (d0 mod 2)>(%b)

      // ----------------------------------------------------------
      // Read one batch from DMA and store it into buffer1
      // ----------------------------------------------------------
      affine.for %i = 0 to 16 {
        %v = dataflow.stream_read %arg1
            : (!dataflow.stream<i32, 16>) -> i32

        memref.store %v, %arg2[%buffer_index, %i]
            : memref<2x16xi32>
      }

      // ----------------------------------------------------------
      // Notify compute which buffer is ready
      // ----------------------------------------------------------
      %buffer_index_i8 =
          arith.index_cast %buffer_index : index to i8

      dataflow.stream_write %arg3, %buffer_index_i8
          : <i8, 2>, i8
    }

    return
  }


  // ------------------------------------------------------------
  // Compute node
  //
  // For each batch:
  //   1. Peek ctrl1 to determine which buffer is ready
  //   2. Read 16 values from buffer1
  //   3. Square them
  //   4. Write results into the other ping-pong buffer
  //   5. Consume ctrl1
  //   6. Send output buffer index through ctrl2
  // ------------------------------------------------------------

  func.func @top_node1(
      %arg0 : !dataflow.stream<i8, 2>,      // ctrl1
      %arg1 : !dataflow.stream<i8, 2>,      // ctrl2
      %arg2 : memref<2x16xi32>,             // buffer1
      %arg3 : memref<2x16xi32>) {           // buffer2

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    affine.for %b = 0 to 8 {

      // ----------------------------------------------------------
      // Wait for / inspect the input buffer index.
      //
      // Equivalent to:
      //
      //     unsigned char buffer_index = ctrl_in.peek();
      // ----------------------------------------------------------
      %buffer_index_i8 = dataflow.stream_peek %arg0
          : (!dataflow.stream<i8, 2>) -> i8

      %buffer_index =
          arith.index_cast %buffer_index_i8 : i8 to index

      // ----------------------------------------------------------
      // Output buffer is the opposite ping-pong bank:
      //
      //     output_buffer_index = 1 - buffer_index
      // ----------------------------------------------------------
      %output_buffer_index =
          arith.subi %c1, %buffer_index : index

      // ----------------------------------------------------------
      // Compute:
      //
      //     buffer2[output_buffer][i] =
      //         buffer1[input_buffer][i] *
      //         buffer1[input_buffer][i]
      // ----------------------------------------------------------
      affine.for %i = 0 to 16 {
        %v = memref.load %arg2[%buffer_index, %i]
            : memref<2x16xi32>

        %squared = arith.muli %v, %v : i32

        memref.store %squared, %arg3[%output_buffer_index, %i]
            : memref<2x16xi32>
      }

      // ----------------------------------------------------------
      // Consume ctrl1 after the entire batch has been processed.
      // ----------------------------------------------------------
      %consumed = dataflow.stream_read %arg0
          : (!dataflow.stream<i8, 2>) -> i8

      // ----------------------------------------------------------
      // Notify store which buffer contains the completed batch.
      // ----------------------------------------------------------
      %output_buffer_i8 =
          arith.index_cast %output_buffer_index : index to i8

      dataflow.stream_write %arg1, %output_buffer_i8
          : <i8, 2>, i8
    }

    return
  }


  // ------------------------------------------------------------
  // Store node
  //
  // For each batch:
  //   1. Peek ctrl2 to determine which output buffer is ready
  //   2. Construct DMA write request:
  //        (16 << 32) | (128 + batch * 16)
  //   3. Send request
  //   4. Write 16 values from buffer2 to DMA
  //   5. Consume ctrl2
  // ------------------------------------------------------------

  func.func @top_node2(
      %arg0 : !dataflow.stream<i8, 2>,      // ctrl2
      %arg1 : memref<2x16xi32>,             // buffer2
      %arg2 : !dataflow.stream<i64, 8>,     // dma_write_ctrl
      %arg3 : !dataflow.stream<i32, 16>) {  // dma_write_data

    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index

    %c32_i64 = arith.constant 32 : i64
    %c16_i64 = arith.constant 16 : i64

    affine.for %b = 0 to 8 {

      // ----------------------------------------------------------
      // Wait for / inspect the completed output buffer.
      // ----------------------------------------------------------
      %buffer_index_i8 = dataflow.stream_peek %arg0
          : (!dataflow.stream<i8, 2>) -> i8

      %buffer_index =
          arith.index_cast %buffer_index_i8 : i8 to index

      // ----------------------------------------------------------
      // output_base = OUT_BASE + b * BATCH_ITEM_NUMBER
      //
      //              = 128 + b * 16
      // ----------------------------------------------------------
      %output_base = affine.apply
          affine_map<(d0) -> (128 + d0 * 16)>(%b)

      // ----------------------------------------------------------
      // Pack DMA write request:
      //
      //     request = (16 << 32) | output_base
      // ----------------------------------------------------------
      %output_base_i64 =
          arith.index_cast %output_base : index to i64

      %len_shifted =
          arith.shli %c16_i64, %c32_i64 : i64

      %req =
          arith.ori %len_shifted, %output_base_i64 : i64

      dataflow.stream_write %arg2, %req
          : <i64, 8>, i64

      // ----------------------------------------------------------
      // Write 16 values from the selected ping-pong buffer.
      // ----------------------------------------------------------
      affine.for %i = 0 to 16 {
        %v = memref.load %arg1[%buffer_index, %i]
            : memref<2x16xi32>

        dataflow.stream_write %arg3, %v
            : <i32, 16>, i32
      }

      // ----------------------------------------------------------
      // Consume ctrl2 after the entire batch has been written.
      // ----------------------------------------------------------
      %consumed = dataflow.stream_read %arg0
          : (!dataflow.stream<i8, 2>) -> i8
    }

    return
  }


  // ------------------------------------------------------------
  // Top-level function
  //
  // This corresponds to:
  //
  //     DATAFLOW
  //     hls::stream<unsigned char> ctrl1, ctrl2;
  //     unsigned int buffer1[32];
  //     unsigned int buffer2[32];
  //
  //     load(...);
  //     compute(...);
  //     store(...);
  // ------------------------------------------------------------

  func.func @top(
      %arg0 : !dataflow.stream<i64, 8>,    // dma_read_ctrl
      %arg1 : !dataflow.stream<i32, 16>,   // dma_read_data
      %arg2 : !dataflow.stream<i64, 8>,    // dma_write_ctrl
      %arg3 : !dataflow.stream<i32, 16>) { // dma_write_data

    // Control streams between pipeline stages.
    %ctrl1 = dataflow.stream {depth = 2 : i32}
        : <i8, 2>

    %ctrl2 = dataflow.stream {depth = 2 : i32}
        : <i8, 2>

    // Double-buffered local memories.
    //
    // These represent:
    //
    //     unsigned int buffer1[32];
    //     unsigned int buffer2[32];
    //
    // as two banks of 16 elements.
    %buffer1 = memref.alloc() : memref<2x16xi32>
    %buffer2 = memref.alloc() : memref<2x16xi32>

    // Load stage.
    call @top_node0(
        %arg0,
        %arg1,
        %buffer1,
        %ctrl1
      ) : (
        !dataflow.stream<i64, 8>,
        !dataflow.stream<i32, 16>,
        memref<2x16xi32>,
        !dataflow.stream<i8, 2>
      ) -> ()

    // Compute stage.
    call @top_node1(
        %ctrl1,
        %ctrl2,
        %buffer1,
        %buffer2
      ) : (
        !dataflow.stream<i8, 2>,
        !dataflow.stream<i8, 2>,
        memref<2x16xi32>,
        memref<2x16xi32>
      ) -> ()

    // Store stage.
    call @top_node2(
        %ctrl2,
        %buffer2,
        %arg2,
        %arg3
      ) : (
        !dataflow.stream<i8, 2>,
        memref<2x16xi32>,
        !dataflow.stream<i64, 8>,
        !dataflow.stream<i32, 16>
      ) -> ()

    return
  }
}
