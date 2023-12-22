/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef DATA_UTILS_H
#define DATA_UTILS_H
#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>
#include <iomanip>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "acl/acl.h"

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO] " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN] " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR] " fmt "\n", ##args)
#define RETURN_IF_NOT_SUCCESS(expr, format, action)  \
    do {                             \
        if ((expr) != ACL_SUCCESS) { \
            ERROR_LOG(format);       \
            action;                  \
        }                            \
    } while (0)

#define CALL_RT(x) \
    if (auto ret = (x) != 0) { \
        std::cout << "[ERROR] Failed to exec acl api " << #x << ", result: " << ret << std::endl; \
        return -1; \
    } else { \
        /*\
        // 正确不需要输出信息\
        std::cout << "[INFO] Succeeded to exec acl api " << #x << std::endl; */\
    }


#define IF_NOT_SUCCESS_RETURN_FALSE(expr, format, action)  \
    do {                             \
        if ((expr) != ACL_SUCCESS) { \
            ERROR_LOG(format);       \
            action;                  \
            return false;            \
        }                            \
    } while (0)

#define ACL_ERROR_LOG(fmt, args...) fprintf(stdout, "[ACL_ERROR]  " fmt "\n", ##args)


/**
 * @brief Get Acl Recent Error Message.
 */
void GetRecentErrMsg()
{
    const char *aclRecentErrMsg = nullptr;
    aclRecentErrMsg = aclGetRecentErrMsg();
    if (aclRecentErrMsg != nullptr) {
        ACL_ERROR_LOG("%s", aclRecentErrMsg);
    } else {
        ACL_ERROR_LOG("Failed to get recent error message.");
    }
}

/**
 * @brief Read data from file
 * @param [in] filePath: file path
 * @param [out] fileSize: file size
 * @return read result
 */
bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("Failed to get file.");
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

/**
 * @brief Write data to file
 * @param [in] filePath: file path
 * @param [in] buffer: data to write to file
 * @param [in] size: size to write
 * @return write result
 */
bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    auto writeSize = write(fd, buffer, size);
    (void) close(fd);
    if (writeSize != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}
#endif // DATA_UTILS_H

/**
 * @brief compare actual data and expected data
 * @param [in] float *actualOutputData: actual data
 * @param [in] float *expectedOutputData: expected data
 * @param [in] uint64_t len: data length
 * @return isCorrect
 */
bool compareFp32OutputData(float *actualOutputData, float *expectedOutputData, uint64_t len)
{
    double error = 0;
    int64_t errorCount = 0;
    int64_t lastError = -1;
    int64_t continuous = 0;
    int64_t maxContinous = 0;
    int64_t i = 0;
    float ratios[] = {0.0001, 0.0001};
    for (i = 0; i < len; i++) {
        float actualOutputItem = *(actualOutputData + i);
        float expectedOutputItem = *(expectedOutputData + i);
        if (i >= 0 && i < 10) {
            printf("our: %f, expected: %f\n", actualOutputItem, expectedOutputItem);
        }
        float tmp = std::abs(std::min(expectedOutputItem, actualOutputItem) * ratios[1]);
        float limitError = tmp;
        if (std::abs(actualOutputItem - expectedOutputItem) > 0.1 && errorCount < 16) {
            std::cout << "sub super 0.1! actual:" << actualOutputItem <<  ", expected:" << expectedOutputItem << std::endl;
        }
        if (std::abs(actualOutputItem - expectedOutputItem) > limitError) {
            errorCount++;
            if (i == lastError + 1) {
                continuous++;
            } else {
                if (maxContinous < continuous) {
                    maxContinous = continuous;
                }
                continuous = 1;
            }
            lastError = i;
        }
        error += std::min((float)1, std::abs((actualOutputItem - expectedOutputItem) / std::min(expectedOutputItem, actualOutputItem)));
    }
    error = error / len;
    std::cout << "error: " << error << std::endl;
    if (i == len - 1) {
        if (maxContinous < continuous) {
            maxContinous = continuous;
        }
    }

    int count = 0;
    if (errorCount >= len * ratios[0] || maxContinous > 16) {
        for (i = 0; i < len; i++) {          
            float actualOutputItem = *(actualOutputData + i);
            float expectedOutputItem = *(expectedOutputData + i);
            float tmp = std::abs(expectedOutputItem * ratios[1]);
            float limitError = tmp;
            if (std::abs(actualOutputItem - expectedOutputItem) > limitError && errorCount < 16) {
                std::cout << "index:" << i << " ,cmprlst:" << std::abs((actualOutputItem - expectedOutputItem) / std::min(expectedOutputItem, actualOutputItem)) <<
                    " ,actualDataf:" << actualOutputItem << " ,expectedDataf:" <<
                    expectedOutputItem << std::endl;
            }
        }
        printf("cmp len:%lu\n",len);
        std::cout << "errorCount:" << errorCount << std::endl;
        printf("ratio:%.10lf\n",(double)errorCount / len);
        return false;
    } else {
        printf("cmp len:%lu\n",len);
        std::cout << "errorCount:" << errorCount << std::endl;
        printf("ratio:%.10lf\n",(double)errorCount / len);
        return true;
    }
}