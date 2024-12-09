Shader "Hidden/TinyTaa"
{
    SubShader
    {
        // Pass 3
        //Cull Off
        //ZWrite On
        Cull Off ZWrite Off ZTest Always
        Pass
        {
            //Name "TinyTaa"

            

            HLSLPROGRAM

            #pragma exclude_renderers d3d11_9x
            #pragma target 4.5

            #pragma vertex vertTaa
            #pragma fragment fragTaa

            // -------------------------------------
            // Includes
            //#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            
            

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/UnityInput.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/GlobalSamplers.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Color.hlsl"
            // -------------------------------------
            // Structs
            struct Attributes
            {
                uint vertexID   : SV_VertexID;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 position : SV_POSITION;
                float2 texcoord : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            // -------------------------------------
            // Vertex
            Varyings vertTaa(Attributes input)
            {
                Varyings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

                // TODO: Use Core Blitter vert.
                output.position = GetFullScreenTriangleVertexPosition(input.vertexID);
                output.texcoord = GetFullScreenTriangleTexCoord(input.vertexID);
                return output;
            }

            // -------------------------------------
            TEXTURE2D_X(_UpscaledTexture);
            TEXTURE2D_X(_LastFrameTexture);
            TEXTURE2D_X(_MotionVectorTexture);

            float4 _UpscaledTexture_TexelSize;
            float4 _LastFrameTexture_TexelSize;
            float _TaaFilterWeights[9];
            //SAMPLER(sampler_ImageBeforeSR_0);
            //SAMPLER(sampler_MotionVectorTexture_1);

            
            float PerceptualWeight(float3 c)
            {

                return rcp(Luminance(c) + 1.0);
            }

            float PerceptualInvWeight(float3 c)
            {
                return rcp(1.0 - Luminance(c));
            }
            float3 WorkingToPerceptual(float3 c)
            {
                float scale = PerceptualWeight(c);
                return c * scale;
            }

            float3 PerceptualToWorking(float3 c)
            {
                float scale = PerceptualInvWeight(c);
                return c * scale;
            }
            half3 PostFxSpaceToLinear(float3 src)
            {
        // gamma 2.0 is a good enough approximation
                return src*src;
            }

            half3 LinearToPostFxSpace(float3 src)
            {
                return sqrt(src);
            }
            half3 SceneToWorkingSpace(half3 src)
            {
                half3 linColor = PostFxSpaceToLinear(src);
        
                half3 dst = linColor;
                return dst;
            }

            half3 WorkingSpaceToScene(half3 src)
            {
                half3 linColor = src;
        
                half3 dst = LinearToPostFxSpace(linColor);
                return dst;
            }
            half3 ApplyHistoryColorLerp(half3 workingAccumColor, half3 workingCenterColor, float t)
            {
                half3 perceptualAccumColor = WorkingToPerceptual(workingAccumColor);
                half3 perceptualCenterColor = WorkingToPerceptual(workingCenterColor);

                half3 perceptualDstColor = lerp(perceptualAccumColor, perceptualCenterColor, t);
                half3 workingDstColor = PerceptualToWorking(perceptualDstColor);

                return workingDstColor;
            }
            half3 SampleBicubic5TapHalf(TEXTURE2D_X(sourceTexture), float2 UV, float4 sourceTexture_TexelSize)
            {
                const float2 sourceTextureSize = sourceTexture_TexelSize.zw;
                const float2 sourceTexelSize = sourceTexture_TexelSize.xy;

                float2 samplePos = UV * sourceTextureSize;
                float2 tc1 = floor(samplePos - 0.5) + 0.5;
                half2 f = samplePos - tc1;
                half2 f2 = f * f;
                half2 f3 = f * f2;

                half c = 0.5;

                half2 w0 = -c         * f3 +  2.0 * c         * f2 - c * f;
                half2 w1 =  (2.0 - c) * f3 - (3.0 - c)        * f2          + 1.0;
                half2 w2 = -(2.0 - c) * f3 + (3.0 - 2.0 * c)  * f2 + c * f;
                half2 w3 = c          * f3 - c                * f2;

                half2 w12 = w1 + w2;
                float2 tc0 = sourceTexelSize  * (tc1 - 1.0);
                float2 tc3 = sourceTexelSize  * (tc1 + 2.0);
                float2 tc12 = sourceTexelSize * (tc1 + w2 / w12);

                /*half3 s0 = SceneToWorkingSpace(SAMPLE_TEXTURE2D_X(sourceTexture, sampler_LinearClamp, float2(tc12.x, tc0.y)).xyz);
                half3 s1 = SceneToWorkingSpace(SAMPLE_TEXTURE2D_X(sourceTexture, sampler_LinearClamp, float2(tc0.x, tc12.y)).xyz);
                half3 s2 = SceneToWorkingSpace(SAMPLE_TEXTURE2D_X(sourceTexture, sampler_LinearClamp, float2(tc12.x, tc12.y)).xyz);
                half3 s3 = SceneToWorkingSpace(SAMPLE_TEXTURE2D_X(sourceTexture, sampler_LinearClamp, float2(tc3.x, tc12.y)).xyz);
                half3 s4 = SceneToWorkingSpace(SAMPLE_TEXTURE2D_X(sourceTexture, sampler_LinearClamp, float2(tc12.x, tc3.y)).xyz);*/

                half3 s0 = SceneToWorkingSpace(SAMPLE_TEXTURE2D_X(sourceTexture, sampler_LinearClamp, float2(tc12.x, tc0.y)).xyz);
                half3 s1 = SceneToWorkingSpace(SAMPLE_TEXTURE2D_X(sourceTexture, sampler_LinearClamp, float2(tc0.x, tc12.y)).xyz);
                half3 s2 = SceneToWorkingSpace(SAMPLE_TEXTURE2D_X(sourceTexture, sampler_LinearClamp, float2(tc12.x, tc12.y)).xyz);
                half3 s3 = SceneToWorkingSpace(SAMPLE_TEXTURE2D_X(sourceTexture, sampler_LinearClamp, float2(tc3.x, tc12.y)).xyz);
                half3 s4 = SceneToWorkingSpace(SAMPLE_TEXTURE2D_X(sourceTexture, sampler_LinearClamp, float2(tc12.x, tc3.y)).xyz);

                half cw0 = (w12.x * w0.y);
                half cw1 = (w0.x * w12.y);
                half cw2 = (w12.x * w12.y);
                half cw3 = (w3.x * w12.y);
                half cw4 = (w12.x *  w3.y);

                s0 *= cw0;
                s1 *= cw1;
                s2 *= cw2;
                s3 *= cw3;
                s4 *= cw4;

                half3 historyFiltered = s0 + s1 + s2 + s3 + s4;
                half weightSum = cw0 + cw1 + cw2 + cw3 + cw4;

                half3 filteredVal = historyFiltered * rcp(weightSum);

                return filteredVal;
            }
            half3 FilterColor(float2 uv, float weights[9])
            {
           half3 filtered = weights[0] * (SAMPLE_TEXTURE2D_X(_UpscaledTexture,sampler_PointClamp,uv+_UpscaledTexture_TexelSize*float2(0.0, 0.0f)).xyz);
                                          
                filtered += weights[1] * (SAMPLE_TEXTURE2D_X(_UpscaledTexture,sampler_PointClamp,uv+_UpscaledTexture_TexelSize*float2(0.0f, 1.0)).xyz);
                filtered += weights[2] * (SAMPLE_TEXTURE2D_X(_UpscaledTexture,sampler_PointClamp,uv+_UpscaledTexture_TexelSize*float2(1.0f, 0.0f)).xyz);
                filtered += weights[3] * (SAMPLE_TEXTURE2D_X(_UpscaledTexture,sampler_PointClamp,uv+_UpscaledTexture_TexelSize*float2(-1.0f, 0.0f)).xyz);
                filtered += weights[4] * (SAMPLE_TEXTURE2D_X(_UpscaledTexture,sampler_PointClamp,uv+_UpscaledTexture_TexelSize*float2(0.0f, -1.0f)).xyz);
                                                             
                filtered += weights[5] * (SAMPLE_TEXTURE2D_X(_UpscaledTexture,sampler_PointClamp,uv+_UpscaledTexture_TexelSize*float2(-1.0f, 1.0f)).xyz);
                filtered += weights[6] * (SAMPLE_TEXTURE2D_X(_UpscaledTexture,sampler_PointClamp,uv+_UpscaledTexture_TexelSize*float2(1.0f, -1.0f)).xyz);
                filtered += weights[7] * (SAMPLE_TEXTURE2D_X(_UpscaledTexture,sampler_PointClamp,uv+_UpscaledTexture_TexelSize*float2(1.0f, 1.0f)).xyz);
                filtered += weights[8] * (SAMPLE_TEXTURE2D_X(_UpscaledTexture,sampler_PointClamp,uv+_UpscaledTexture_TexelSize*float2(-1.0f, -1.0f)).xyz);
                
                    return filtered;
                
            }
            //hgzadd
            float HaltonSequence(float i, float prime) 
            {
                float f = 1.0;
                float r = 0.0;
    
                // Ensure i is positive and large enough
                i = i + 1.0; // Add a small offset to avoid i being too small
    
                while (i > 0.0) 
                {
                    f = f / prime;
                    r = r + f * i % prime;
                    i = floor(i / prime);
                }
                return r;
            }

            half2 GetLowDiscrepancyJitter(int frameID) 
            {
                float2 jitter;
    
                // Use larger frameID or an offset for jitter
                jitter.x = HaltonSequence(frameID + 100, 2.0);  // Offset frameID to avoid too small values
                jitter.y = HaltonSequence(frameID + 200, 3.0);  // Use different primes for more variation
    
                // Return the jitter, can scale or shift it later for UV range
                return jitter * 0.5 - 0.25;
            }

            void AdjustColorBox(inout half3 boxMin, inout half3 boxMax,  float2 uv, half currX, half currY)
            {
                half3 color = SAMPLE_TEXTURE2D_X(_UpscaledTexture, sampler_PointClamp, uv + _UpscaledTexture_TexelSize.xy * float2(currX, currY)).xyz;
                boxMin = min(color, boxMin);
                boxMax = max(color, boxMax);
            }

            float _ScreenWidth;
            float _ScreenHeight;
            // Fragment
            /*half4 fragTaa(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

                float2 uv = UnityStereoTransformScreenSpaceTex(input.texcoord);
                float2 mv = SAMPLE_TEXTURE2D_X(_MotionVectorTexture, sampler_PointClamp, uv).xy;
                
                float2 preUV = uv - mv;
                half3 preColor = SAMPLE_TEXTURE2D_X(_LastFrameTexture, sampler_LinearClamp, preUV).xyz;
                half3 color = SAMPLE_TEXTURE2D_X(_UpscaledTexture, sampler_PointClamp, uv).xyz;

                
                return half4(preColor*0.9+color*0.1,1.0);
                
            }*/

            half4 fragTaa(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

                float2 uv = UnityStereoTransformScreenSpaceTex(input.texcoord);
                float2 mv = SAMPLE_TEXTURE2D_X(_MotionVectorTexture, sampler_PointClamp, uv).xy;
                /*int frameID = _Time.y;
                float2 jitter = GetLowDiscrepancyJitter(frameID);
                jitter.x /= _ScreenWidth;
                jitter.y /= _ScreenHeight;
                float2 preUV = uv - mv + jitter;*/
                float2 preUV = uv - mv;
                //half3 preColor = SampleBicubic5TapHalf(_LastFrameTexture, preUV, _LastFrameTexture_TexelSize.xyzw) ;
                //half3 color =FilterColor(uv, _TaaFilterWeights);
                half3 preColor = SAMPLE_TEXTURE2D_X(_LastFrameTexture, sampler_PointClamp, preUV).xyz;
                half3 color = SAMPLE_TEXTURE2D_X(_UpscaledTexture, sampler_PointClamp, uv).xyz;

                half3 boxMax = color;
                half3 boxMin = color;
                
                
                AdjustColorBox(boxMin, boxMax, uv, 0.0f, -1.0f);
                AdjustColorBox(boxMin, boxMax, uv, -1.0f, 0.0f);
                AdjustColorBox(boxMin, boxMax, uv, 1.0f, 0.0f);
                AdjustColorBox(boxMin, boxMax, uv, 0.0f, 1.0f);
                half3 clampAccum=clamp(preColor, boxMin, boxMax);
                return half4(clampAccum*0.9+color*0.1,1.0);
                //return half4(color,1.0);

                //return color;
            }

            /*half4 fragTaa(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

                float2 uv = UnityStereoTransformScreenSpaceTex(input.texcoord);
                float2 mv = SAMPLE_TEXTURE2D_X(_MotionVectorTexture, sampler_LinearClamp, uv).xy;

                float2 preUV = uv - mv;
                half3 preColor = SceneToWorkingSpace(SAMPLE_TEXTURE2D_X(_LastFrameTexture, sampler_PointClamp, uv).xyz);
                half3 color = SceneToWorkingSpace(SAMPLE_TEXTURE2D_X(_UpscaledTexture, sampler_PointClamp, uv).xyz);
                
                half3 workingColor = ApplyHistoryColorLerp(preColor, color, 0.1f);
                half3 dstSceneColor = WorkingSpaceToScene(workingColor);
                return half4(max(dstSceneColor, 0.0), 1.0);
                //return half4(preColor*0.5+color*0.5,1.0);
                //return half4(color,1.0);

                //return color;
            }*/
            ENDHLSL
        }
    }
}