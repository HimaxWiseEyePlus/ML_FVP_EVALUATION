/*
 * Copyright (c) 2021 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef AD_MODEL_HPP
#define AD_MODEL_HPP

#include "Model.hpp"
#include <cstdint>
extern const int g_FrameLength;
extern const int g_FrameStride;
extern const float g_ScoreThreshold;
extern const float g_TrainingMean;

namespace arm {
namespace app {

    class AdModel : public Model {
    protected:
        /** @brief   Gets the reference to op resolver interface class */
        const tflite::MicroOpResolver& GetOpResolver() override;

        /** @brief   Adds operations to the op resolver instance */
        bool EnlistOperations() override;

        const uint8_t* ModelPointer() override;

        size_t ModelSize() override;

    private:
        /* Maximum number of individual operations that can be enlisted */
        static constexpr int ms_maxOpCnt = 6;

        /* A mutable op resolver instance */
        tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
    };

} /* namespace app */
} /* namespace arm */

#endif /* AD_MODEL_HPP */
