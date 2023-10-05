// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef YOLO_H
#define YOLO_H

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>
#include <utility>
#include <net.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <unordered_map>
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};
struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};
struct ObjectInfo {
    std::string label;
    int lastSeenFrame;
};
class Yolo
{
public:
    Yolo();

    // int load(const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);
    int loadMFnet(AAssetManager* mgr, const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);
    int load(AAssetManager* mgr, const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);
    int detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.2f, float nms_threshold = 0.5f);
    int draw(cv::Mat& rgb, const std::vector<Object>& objects);

    // std::pair<std::vector<std::vector<float>>, std::vector<std::string>> load_feature_db_txt(const std::string& str, const std::string& delimiter)
    // {
    //     std::vector<std::vector<float>> featureVectorsTemp;
    //     std::vector<std::string> labelsTemp;

    //     std::string::size_type pos = 0;
    //     std::string::size_type prev = 0;
    //     while ((pos = str.find(delimiter, prev)) != std::string::npos)
    //     {
    //         std::string line = str.substr(prev, pos - prev);
    //         std::istringstream iss(line);
    //         std::string featureStr, label;
    //         std::getline(iss, featureStr, ':');
    //         featureStr = featureStr.substr(1, featureStr.size() - 2);
    //         std::getline(iss, label, ':');
    //         label = label.substr(2, label.size() - 3);

    //         std::vector<float> featureVec;
    //         std::istringstream featureStream(featureStr);
    //         std::string val;
    //         while (std::getline(featureStream, val, ' ')) {
    //             if (!val.empty()){
    //                 featureVec.push_back(std::stof(val));
    //             }
    //         }
    //         featureVectorsTemp.push_back(featureVec);
    //         labelsTemp.push_back(label);
    //         prev = pos + delimiter.size();

    //     }

    //     // To get the last substring (or only, if delimiter is not found)
    //     return {featureVectorsTemp, labelsTemp};
    // }
// std::vector<float> convert_to_vector(const ncnn::Mat& mat) {
//     std::vector<float> vec;
//     int channels = mat.c;
//     int width = mat.w;
//     int height = mat.h;

//     vec.reserve(channels * width * height);

//     for (int c = 0; c < channels; ++c) {
//         for (int h = 0; h < height; ++h) {
//             for (int w = 0; w < width; ++w) {
//                 float value = mat.channel(c).row(h)[w];
//                 vec.push_back(value);
//             }
//         }
//     }

//     return vec;
// }

    // int findMostSimilar(const std::vector<float>& V1) {
    //     Eigen::VectorXf similarities = batchCosineSimilarity(V1);

    //     Eigen::Index maxIndex;
    //     float maxSimilarity = similarities.maxCoeff(&maxIndex);
    //     if (maxSimilarity < 0.34) {
    //         return -1;
    //     }
    //     return static_cast<int>(maxIndex);
    // }

    // Eigen::VectorXf batchCosineSimilarity(const std::vector<float>& A) {

    //     Eigen::VectorXf vecA = Eigen::VectorXf::Map(A.data(), A.size());
    //     vecA.normalize();
    //     return B_normalized * vecA;
    // }
private:
    ncnn::Net yolo;
    // Tracker tracker;
    int frame_index = 0;
    int frameThreshold = 60;
    std::vector<cv::Rect> bbox_per_frame;
    std::unordered_map<int, ObjectInfo> idInfoMap;
    int target_size;
    float mean_vals[3];
    float norm_vals[3];
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
    // std::vector<std::vector<float>> featureVectors;
    // std::vector<std::string> labels;
    Eigen::MatrixXf matB;
    // Eigen::MatrixXf B_normalized;
    // SqueezNet INT
    ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    ncnn::PoolAllocator g_workspace_pool_allocator;

    // std::vector<std::string> squeezenet_words;
    ncnn::Net MFnet;
    ncnn::Option opt;
};

#endif // NANODET_H
