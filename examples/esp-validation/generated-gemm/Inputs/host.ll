; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @esp_accel_wait()

declare void @esp_accel_start()

declare void @esp_copy_from_shared_f32(i64, i64, i64, ptr)

declare void @esp_copy_to_shared_f32(i64, ptr, i64, i64)

declare void @esp_free_shared(i64)

declare i64 @esp_alloc_shared(i64)

define void @forward_kernel(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5) {
  %7 = insertvalue { ptr, ptr, i64 } undef, ptr %5, 0
  %8 = insertvalue { ptr, ptr, i64 } %7, ptr %5, 1
  %9 = insertvalue { ptr, ptr, i64 } %8, i64 0, 2
  %10 = insertvalue { ptr, ptr, i64 } undef, ptr %4, 0
  %11 = insertvalue { ptr, ptr, i64 } %10, ptr %4, 1
  %12 = insertvalue { ptr, ptr, i64 } %11, i64 0, 2
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %3, 0
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, ptr %3, 1
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, i64 0, 2
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, i64 8, 3, 0
  %17 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, i64 12, 4, 0
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, i64 12, 3, 1
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 1, 4, 1
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %2, 0
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, ptr %2, 1
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, i64 0, 2
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, i64 16, 3, 0
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 12, 4, 0
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 12, 3, 1
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 1, 4, 1
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, ptr %1, 1
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, i64 0, 2
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 8, 3, 0
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 16, 4, 0
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 16, 3, 1
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 1, 4, 1
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %0, 0
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, ptr %0, 1
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 0, 2
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 8, 3, 0
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 12, 4, 0
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 12, 3, 1
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 1, 4, 1
  %41 = call i64 @esp_alloc_shared(i64 2056)
  %42 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, ptr %42, align 8
  %43 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %42, 1
  %44 = extractvalue { i64, ptr } %43, 0
  %45 = extractvalue { i64, ptr } %43, 1
  call void @esp_copy_to_shared_f32(i64 %44, ptr %45, i64 %41, i64 0)
  %46 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, ptr %46, align 8
  %47 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %46, 1
  %48 = extractvalue { i64, ptr } %47, 0
  %49 = extractvalue { i64, ptr } %47, 1
  call void @esp_copy_to_shared_f32(i64 %48, ptr %49, i64 %41, i64 96)
  %50 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, ptr %50, align 8
  %51 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %50, 1
  %52 = extractvalue { i64, ptr } %51, 0
  %53 = extractvalue { i64, ptr } %51, 1
  call void @esp_copy_to_shared_f32(i64 %52, ptr %53, i64 %41, i64 224)
  %54 = alloca { ptr, ptr, i64 }, i64 1, align 8
  store { ptr, ptr, i64 } %12, ptr %54, align 8
  %55 = insertvalue { i64, ptr } { i64 0, ptr undef }, ptr %54, 1
  %56 = extractvalue { i64, ptr } %55, 0
  %57 = extractvalue { i64, ptr } %55, 1
  call void @esp_copy_to_shared_f32(i64 %56, ptr %57, i64 %41, i64 512)
  %58 = alloca { ptr, ptr, i64 }, i64 1, align 8
  store { ptr, ptr, i64 } %9, ptr %58, align 8
  %59 = insertvalue { i64, ptr } { i64 0, ptr undef }, ptr %58, 1
  %60 = extractvalue { i64, ptr } %59, 0
  %61 = extractvalue { i64, ptr } %59, 1
  call void @esp_copy_to_shared_f32(i64 %60, ptr %61, i64 %41, i64 513)
  call void @esp_accel_start()
  call void @esp_accel_wait()
  %62 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, ptr %62, align 8
  %63 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %62, 1
  %64 = extractvalue { i64, ptr } %63, 0
  %65 = extractvalue { i64, ptr } %63, 1
  call void @esp_copy_from_shared_f32(i64 %41, i64 416, i64 %64, ptr %65)
  call void @esp_free_shared(i64 %41)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
