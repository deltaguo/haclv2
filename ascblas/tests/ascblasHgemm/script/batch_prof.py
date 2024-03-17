import os
import csv
import subprocess
import sys
import re
kernel_name = "hgemm"
device_id = 0
start_idx = 0

def run_profiling(command):
    for _ in range(3):  # Retry up to 3 times
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout.strip().split("time_us: ")

        if len(output) < 2:
            # Handle the case where the expected output is not present
            print(f"Error: Unexpected output format - retry")
        else:
            return float(re.findall(r"[-+]?\d*\.\d+|\d+", output[1])[0]) / 1000

    # Return -1 if unable to get valid ascblasTime after 3 retries
    print(f"Error: Unable to retrieve valid ascblasTime for command - {command}")
    return -1

def batch_prof(kernel_name, device_id):
    if "result" in os.getcwd():
        params_filepath = "../result/" + kernel_name + "_data.csv"
        accur_output_filepath = "../result/prof_data/" + kernel_name + "_prof.csv"
    else:
        params_filepath = "result/" + kernel_name + "_data.csv"
        accur_output_filepath = "result/prof_data/" + kernel_name + "_prof.csv"

    fieldnames = [
        "transA",
        "transB",
        "M",
        "N",
        "K",
        "lda",
        "ldb",
        "ldc",
        "ascblasTime(ms)",
        "Tflops",
        "utilization ratio"
    ]

    with open(accur_output_filepath, "a+", newline="") as output_csvfile:
        writer = csv.writer(output_csvfile)
        writer.writerow(fieldnames)

    with open(params_filepath) as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            if index < start_idx:  # Skip header row
                continue

            transA = row["transA"]
            transB = row["transB"]
            M = int(row["M"])
            N = int(row["N"])
            K = int(row["K"])
            lda = int(row["lda"])
            ldb = int(row["ldb"])
            ldc = int(row["ldc"])

            if "result" in os.getcwd():
                command = f"bash ../run.sh {transA} {transB} {M} {N} {K} {lda} {ldb} {ldc} prof {device_id}"
            else:
                command = f"bash ./run.sh {transA} {transB} {M} {N} {K} {lda} {ldb} {ldc} prof {device_id}"

            ascblasTime = run_profiling(command)

            Tflops = 2 * M * N * K * 1e-9 / ascblasTime
            with open(accur_output_filepath, "a+", newline="") as output_csvfile:
                writer = csv.DictWriter(output_csvfile, fieldnames)
                writer.writerow({
                    "transA": transA,
                    "transB": transB,
                    "M": M,
                    "N": N,
                    "K": K,
                    "lda": lda,
                    "ldb": ldb,
                    "ldc": ldc,
                    "ascblasTime(ms)": ascblasTime,
                    "Tflops": Tflops,
                    "utilization ratio": Tflops / 245.76 if ascblasTime != -1 else -1
                })

            print(transA, transB, M, N, K, lda, ldb, ldc, ascblasTime, Tflops, Tflops / 245.76)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        kernel_name = sys.argv[1]
    if len(sys.argv) > 2:
        device_id = sys.argv[2]
    if len(sys.argv) > 3:
        start_idx = int(sys.argv[3])

    batch_prof(kernel_name, device_id)
