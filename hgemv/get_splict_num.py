core_num=40
result_size=2048
result_block_size=128
result_block_num=(result_size-1)//result_block_size+1
splict_num=1
max_splict_num=4
max_remain = 1
x = 0
if splict_num * result_block_num < core_num // 2:
    for i in range(max_splict_num):
        block_num = (i+1) * result_block_num
        remain  = block_num % core_num
        if remain > max_remain:
            max_remain = remain
            splict_num = i+1
        else:
            break

print(splict_num)