from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from std.sys import argv
from std.testing import assert_equal

# ANCHOR: conv_1d_simple
comptime TPB = 8
comptime SIZE = 6
comptime CONV = 3
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32
comptime in_layout = Layout.row_major(SIZE)
comptime out_layout = Layout.row_major(SIZE)
comptime conv_layout = Layout.row_major(CONV)


def conv_1d_simple[
    in_layout: Layout, out_layout: Layout, conv_layout: Layout
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    b: LayoutTensor[dtype, conv_layout, ImmutAnyOrigin],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var local_i = Int(thread_idx.x)
       
#    if(global_i == 0):
#        for i in range(SIZE):
#            output[i] = 0
#            for j in range(CONV):
#                if i + j < SIZE:
#                    output[i] += a[i + j] * b[j]

#    i = Int(global_i)
#    output[i] = 0
#    for j in range(CONV):
#        if i + j < SIZE:
#            output[i] += a[i + j] * b[j]

    # Allocate shared memory using tensor builder
    var shared_a = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var shared_b = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    if global_i < SIZE:
        shared_a[local_i] = a[global_i]

    if global_i < CONV:
        shared_b[local_i] = b[global_i]

    i = Int(global_i)
    if i < SIZE:
        var local_output : output.element_type = 0
        # unroll loop
        comptime for j in range(CONV):
            if i + j < SIZE:
                local_output += shared_a[i + j] * shared_b[j]
        output[global_i] = local_output


# ANCHOR_END: conv_1d_simple

# ANCHOR: conv_1d_block_boundary
comptime SIZE_2 = 15
comptime CONV_2 = 4
comptime BLOCKS_PER_GRID_2 = (2, 1)
comptime THREADS_PER_BLOCK_2 = (TPB, 1)
comptime in_2_layout = Layout.row_major(SIZE_2)
comptime out_2_layout = Layout.row_major(SIZE_2)
comptime conv_2_layout = Layout.row_major(CONV_2)


def conv_1d_block_boundary[
    in_layout: Layout, out_layout: Layout, conv_layout: Layout, dtype: DType
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    b: LayoutTensor[dtype, conv_layout, ImmutAnyOrigin],
):
    var global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var local_i = Int(thread_idx.x)
    
    var shared_a = LayoutTensor[
        dtype,
        Layout.row_major(TPB + CONV_2 - 1), # the overlap region: CONV_2 - 1
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var shared_b = LayoutTensor[
        dtype,
        Layout.row_major(CONV_2), # convolution mask: CONV_2
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # Copy base data
    if local_i < SIZE_2:
        shared_a[local_i] = a[global_i]
    else:
        shared_a[local_i] = 0

    barrier()

    # Boundary data in the overlap region
    # The same global_i was used to copy the start of the base data (CONV_2 - 1) earlier and is now used to copy the overlap data (CONV_2 - 1)
    if local_i < CONV_2 - 1:
        next_idx = global_i + TPB
        if next_idx < SIZE_2:
            shared_a[TPB + local_i] = a[next_idx]
        else:
            # Initialize out-of-bounds elements to 0 to avoid reading from uninitialized memory
            # which is an undefined behavior
            shared_a[TPB + local_i] = 0

    # copy convolution kernel to sm
    if local_i < CONV_2:
        shared_b[local_i] = b[local_i]

    if global_i < SIZE_2:
        var local_output : output.element_type = 0
        # unroll loop
        comptime for j in range(CONV_2):
            if global_i + j < SIZE_2:
                local_output += shared_a[local_i + j] * shared_b[j]
        output[global_i] = local_output



# ANCHOR_END: conv_1d_block_boundary


def main() raises:
    with DeviceContext() as ctx:
        var size = SIZE_2 if argv()[1] == "--block-boundary" else SIZE
        var conv = CONV_2 if argv()[1] == "--block-boundary" else CONV
        var out = ctx.enqueue_create_buffer[dtype](size)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](size)
        a.enqueue_fill(0)
        var b = ctx.enqueue_create_buffer[dtype](conv)
        b.enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(size):
                a_host[i] = Float32(i)

        with b.map_to_host() as b_host:
            for i in range(conv):
                b_host[i] = Float32(i)

        if len(argv()) != 2 or argv()[1] not in [
            "--simple",
            "--block-boundary",
        ]:
            raise Error(
                "Expected one command-line argument: '--simple' or"
                " '--block-boundary'"
            )

        if argv()[1] == "--simple":
            var out_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](out)
            var a_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](a)
            var b_tensor = LayoutTensor[dtype, conv_layout, ImmutAnyOrigin](b)
            comptime kernel = conv_1d_simple[in_layout, out_layout, conv_layout]
            ctx.enqueue_function[kernel, kernel](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        else:
            var out_tensor = LayoutTensor[dtype, out_2_layout, MutAnyOrigin](
                out
            )
            var a_tensor = LayoutTensor[dtype, in_2_layout, ImmutAnyOrigin](a)
            var b_tensor = LayoutTensor[dtype, conv_2_layout, ImmutAnyOrigin](b)
            comptime kernel = conv_1d_block_boundary[
                in_2_layout, out_2_layout, conv_2_layout, dtype
            ]
            ctx.enqueue_function[kernel, kernel](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )

        ctx.synchronize()
        var expected = ctx.enqueue_create_host_buffer[dtype](size)
        expected.enqueue_fill(0)

        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(size):
                for j in range(conv):
                    if i + j < size:
                        expected[i] += a_host[i + j] * b_host[j]

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(size):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 13 complete ✅")
