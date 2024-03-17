import os
import csv
import subprocess
import sys
import numpy as np

kernel_name = "cgemm"
device_id = 0
start_idx = 0

def batch_error(kernel_name, device_id):
    if ("result" in os.getcwd()):
        params_filepath = "../result/" + kernel_name + "_data.csv"
        accur_output_filepath = "../result/error_data/" + kernel_name + "_error.csv"
    else:
        params_filepath = "result/" + kernel_name + "_data.csv"
        accur_output_filepath = "result/error_data/" + kernel_name + "_error.csv"

    fieldnames = [
        "transA",
        "transB",
        "M",
        "N",
        "K",
        "lda",
        "ldb",
        "ldc",
        "max_abs_error",
        "max_rel_error",
        "avg_abs_error",
        "avg_rel_error",
        "errorCount",
        "errorCountRatio",
    ]
    with open(accur_output_filepath, "a+", newline="") as output_csvfile:
        writer = csv.writer(output_csvfile)
        writer.writerow(fieldnames)
    with open(params_filepath) as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            if index < start_idx:  # 跳过标题行
                continue
            # print(index, row)
            transA = row["transA"]
            transB = row["transB"]
            M = row["M"]
            N = row["N"]
            K = row["K"]
            lda = row["lda"]
            ldb = row["ldb"]
            ldc = row["ldc"]
            if ("result" in os.getcwd()):
                command = f"bash ../run.sh {transA} {transB} {M} {N} {K} {lda} {ldb} {ldc} error {device_id}"
            else:
                command = f"bash ./run.sh {transA} {transB} {M} {N} {K} {lda} {ldb} {ldc} error {device_id}"
            # 执行命令
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            output = result.stdout.strip().split("error_static: ")
            output = output[1].split(", ")
            output = np.pad(output, (0, 6 - len(output)), 'constant', constant_values=(-1,-1))
            if (len(output) != 6):
                output[0] = -1
                output[1] = -1
                output[2] = -1
                output[3] = -1
                output[4] = -1
                output[5] = -1
            # print(result.stdout.strip())
            with open(accur_output_filepath, "a+", newline="") as output_csvfile:
                writer = csv.DictWriter(output_csvfile, fieldnames)
                writer.writerow(
                    {
                        "transA": transA,
                        "transB": transB,
                        "M": M,
                        "N": N,
                        "K": K,
                        "lda": lda,
                        "ldb": ldb,
                        "ldc": ldc,
                        "max_abs_error": output[0],
                        "max_rel_error": output[1],
                        "avg_abs_error": output[2],
                        "avg_rel_error": output[3],
                        "errorCount": output[4],
                        "errorCountRatio": output[5],
                    }
                )
            print(transA, transB, M, N, K, lda, ldb, ldc, output[0], output[1], output[2], output[3], output[4], output[5])

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        kernel_name = sys.argv[1]
    if (len(sys.argv) > 2):
        device_id = sys.argv[2]
    if (len(sys.argv) > 3):
        start_idx = int(sys.argv[3])
    batch_error(kernel_name, device_id)