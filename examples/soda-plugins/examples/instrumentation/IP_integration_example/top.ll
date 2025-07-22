; ModuleID = 'top.c'
source_filename = "top.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.module1_output_t = type { i64, i64, i16, i32 }
%struct.module2_output_t = type { i64, i64, i16 }

@my_ip.module1_output = internal global %struct.module1_output_t zeroinitializer, align 8
@my_ip.module2_output = internal global %struct.module2_output_t zeroinitializer, align 8

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @my_ip(i8 noundef zeroext %0, i32 noundef %1, i32 noundef %2) #0 {
  %4 = alloca i8, align 1
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i8 %0, ptr %4, align 1
  store i32 %1, ptr %5, align 4
  store i32 %2, ptr %6, align 4
  %7 = load i8, ptr %4, align 1
  %8 = zext i8 %7 to i32
  switch i32 %8, label %25 [
    i32 0, label %9
    i32 1, label %14
    i32 2, label %16
    i32 3, label %21
  ]

9:                                                ; preds = %3
  %10 = load i32, ptr %5, align 4
  %11 = load i32, ptr %6, align 4
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i16
  call void @module1(i32 noundef %10, i16 noundef zeroext %13, ptr noundef @my_ip.module1_output)
  br label %26

14:                                               ; preds = %3
  %15 = load i32, ptr %5, align 4
  call void @module2(i32 noundef %15, ptr noundef @my_ip.module2_output)
  br label %26

16:                                               ; preds = %3
  %17 = load i64, ptr @my_ip.module1_output, align 8
  %18 = load i64, ptr getelementptr inbounds (%struct.module1_output_t, ptr @my_ip.module1_output, i32 0, i32 1), align 8
  %19 = load i16, ptr getelementptr inbounds (%struct.module1_output_t, ptr @my_ip.module1_output, i32 0, i32 2), align 8
  %20 = load i32, ptr getelementptr inbounds (%struct.module1_output_t, ptr @my_ip.module1_output, i32 0, i32 3), align 4
  call void @printer1(i64 noundef %17, i64 noundef %18, i16 noundef zeroext %19, i32 noundef %20)
  br label %26

21:                                               ; preds = %3
  %22 = load i64, ptr @my_ip.module2_output, align 8
  %23 = load i64, ptr getelementptr inbounds (%struct.module2_output_t, ptr @my_ip.module2_output, i32 0, i32 1), align 8
  %24 = load i16, ptr getelementptr inbounds (%struct.module2_output_t, ptr @my_ip.module2_output, i32 0, i32 2), align 8
  call void @printer2(i64 noundef %22, i64 noundef %23, i16 noundef zeroext %24)
  br label %26

25:                                               ; preds = %3
  br label %26

26:                                               ; preds = %25, %21, %16, %14, %9
  call void @sodaInstrAssertLessThen(i64 noundef 32, i64 noundef 4)
  ret void
}

declare void @module1(i32 noundef, i16 noundef zeroext, ptr noundef) #1

declare void @module2(i32 noundef, ptr noundef) #1

declare void @printer1(i64 noundef, i64 noundef, i16 noundef zeroext, i32 noundef) #1

declare void @printer2(i64 noundef, i64 noundef, i16 noundef zeroext) #1

declare void @sodaInstrAssertLessThen(i64 noundef, i64 noundef) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"Debian clang version 16.0.6 (15~deb12u1)"}
