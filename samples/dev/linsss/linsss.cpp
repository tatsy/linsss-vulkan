/* Copyright (c) 2019-2020, Sascha Willems
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
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

/*
 * Texture loading (and display) example (including mip maps)
 */

#include "linsss.h"

#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stdexcept>
#include <tinyply.h>

#include <glm/gtx/string_cast.hpp>

#include "gauss.h"

static constexpr uint32_t SHADOW_MAP_SIZE    = 2048;
static constexpr uint32_t MAX_MIP_LEVELS     = 16;
static constexpr float    ENVMAP_SCALE       = 2.0f;
static constexpr int      TSM_UPSAMPLE_RATIO = 4;

LinSSScatter::LinSSScatter()
{
    default_clear_color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    zoom                = -3.5f;
    rotation            = {180.0f, 0.0f, 0.0f};
    title               = "LinSSS";
    name                = "LinSSS";
}

LinSSScatter::~LinSSScatter()
{
    if (device)
    {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        vkDestroyPipeline(get_device().get_handle(), pipelines.light_pass, nullptr);
        vkDestroyPipeline(get_device().get_handle(), pipelines.direct_pass, nullptr);
        vkDestroyPipeline(get_device().get_handle(), pipelines.gauss_filter, nullptr);
        vkDestroyPipeline(get_device().get_handle(), pipelines.linsss, nullptr);
        vkDestroyPipeline(get_device().get_handle(), pipelines.trans_sm, nullptr);
        vkDestroyPipeline(get_device().get_handle(), pipelines.background, nullptr);
        vkDestroyPipeline(get_device().get_handle(), pipelines.deferred, nullptr);
        vkDestroyPipeline(get_device().get_handle(), pipelines.postprocess, nullptr);

        vkDestroyDescriptorPool(get_device().get_handle(), descriptor_pools.light_pass, nullptr);
        vkDestroyDescriptorPool(get_device().get_handle(), descriptor_pools.direct_pass, nullptr);
        vkDestroyDescriptorPool(get_device().get_handle(), descriptor_pools.gauss_filter, nullptr);
        vkDestroyDescriptorPool(get_device().get_handle(), descriptor_pools.linsss, nullptr);
        vkDestroyDescriptorPool(get_device().get_handle(), descriptor_pools.trans_sm, nullptr);
        vkDestroyDescriptorPool(get_device().get_handle(), descriptor_pools.deferred, nullptr);
        vkDestroyDescriptorPool(get_device().get_handle(), descriptor_pools.postprocess, nullptr);

        vkDestroyPipelineLayout(get_device().get_handle(), pipeline_layouts.light_pass, nullptr);
        vkDestroyPipelineLayout(get_device().get_handle(), pipeline_layouts.direct_pass, nullptr);
        vkDestroyPipelineLayout(get_device().get_handle(), pipeline_layouts.gauss_filter, nullptr);
        vkDestroyPipelineLayout(get_device().get_handle(), pipeline_layouts.linsss, nullptr);
        vkDestroyPipelineLayout(get_device().get_handle(), pipeline_layouts.trans_sm, nullptr);
        vkDestroyPipelineLayout(get_device().get_handle(), pipeline_layouts.deferred, nullptr);
        vkDestroyPipelineLayout(get_device().get_handle(), pipeline_layouts.postprocess, nullptr);

        vkDestroyDescriptorSetLayout(get_device().get_handle(), descriptor_set_layouts.light_pass, nullptr);
        vkDestroyDescriptorSetLayout(get_device().get_handle(), descriptor_set_layouts.direct_pass, nullptr);
        vkDestroyDescriptorSetLayout(get_device().get_handle(), descriptor_set_layouts.gauss_filter, nullptr);
        vkDestroyDescriptorSetLayout(get_device().get_handle(), descriptor_set_layouts.linsss, nullptr);
        vkDestroyDescriptorSetLayout(get_device().get_handle(), descriptor_set_layouts.trans_sm, nullptr);
        vkDestroyDescriptorSetLayout(get_device().get_handle(), descriptor_set_layouts.deferred, nullptr);
        vkDestroyDescriptorSetLayout(get_device().get_handle(), descriptor_set_layouts.postprocess, nullptr);

        destroy_custom_framebuffers();
        destroy_custom_render_passes();
    }

    destroy_texture(Ks_texture);
    destroy_texture(envmap_texture);
    destroy_bssrdf(bssrdf);

    model.vertex_buffer.reset();
    model.index_buffer.reset();
    rect.vertex_buffer.reset();
    rect.index_buffer.reset();
    cube.vertex_buffer.reset();
    cube.index_buffer.reset();
    uniform_buffer_vs.reset();
    uniform_buffer_fs.reset();
}

void LinSSScatter::setup_custom_render_passes()
{
    // Setup additional render pass (light pass)
    {
        std::array<VkAttachmentDescription, 4> attachments = {};
        // Color attachment #1
        attachments[0].format         = VK_FORMAT_R32G32B32A32_SFLOAT;
        attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // Color attachment #2
        attachments[1].format         = VK_FORMAT_R32G32B32A32_SFLOAT;
        attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // Color attachment #3
        attachments[2].format         = VK_FORMAT_R32G32B32A32_SFLOAT;
        attachments[2].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[2].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[2].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[2].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[2].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[2].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // Depth attachment
        attachments[3].format         = VK_FORMAT_D32_SFLOAT;
        attachments[3].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[3].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[3].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[3].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[3].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[3].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        std::array<VkAttachmentReference, 3> color_references = {};
        color_references[0].attachment                        = 0;
        color_references[0].layout                            = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_references[1].attachment                        = 1;
        color_references[1].layout                            = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_references[2].attachment                        = 2;
        color_references[2].layout                            = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depth_reference = {};
        depth_reference.attachment            = 3;
        depth_reference.layout                = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass_description    = {};
        subpass_description.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass_description.colorAttachmentCount    = color_references.size();
        subpass_description.pColorAttachments       = color_references.data();
        subpass_description.pDepthStencilAttachment = &depth_reference;
        subpass_description.inputAttachmentCount    = 0;
        subpass_description.pInputAttachments       = nullptr;
        subpass_description.preserveAttachmentCount = 0;
        subpass_description.pPreserveAttachments    = nullptr;
        subpass_description.pResolveAttachments     = nullptr;

        // Subpass dependencies for layout transitions
        std::array<VkSubpassDependency, 2> dependencies;

        dependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass      = 0;
        dependencies[0].srcStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        dependencies[1].srcSubpass      = 0;
        dependencies[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        VkRenderPassCreateInfo render_pass_create_info = {};
        render_pass_create_info.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_create_info.attachmentCount        = static_cast<uint32_t>(attachments.size());
        render_pass_create_info.pAttachments           = attachments.data();
        render_pass_create_info.subpassCount           = 1;
        render_pass_create_info.pSubpasses             = &subpass_description;
        render_pass_create_info.dependencyCount        = static_cast<uint32_t>(dependencies.size());
        render_pass_create_info.pDependencies          = dependencies.data();

        VK_CHECK(vkCreateRenderPass(get_device().get_handle(), &render_pass_create_info, nullptr, &render_passes.light_pass));
    }

    // Setup additional render pass (direct)
    {
        std::array<VkAttachmentDescription, 6> attachments = {};
        // Color attachment #1 (diffuse)
        attachments[0].format         = VK_FORMAT_R32G32B32A32_SFLOAT;
        attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // Color attachment #2 (specular)
        attachments[1].format         = VK_FORMAT_R32G32B32A32_SFLOAT;
        attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // Color attachment #3 (position)
        attachments[2].format         = VK_FORMAT_R32G32B32A32_SFLOAT;
        attachments[2].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[2].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[2].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[2].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[2].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[2].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // Color attachment #4 (normal)
        attachments[3].format         = VK_FORMAT_R32G32B32A32_SFLOAT;
        attachments[3].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[3].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[3].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[3].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[3].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[3].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // Color attachment #5 (depth)
        attachments[4].format         = VK_FORMAT_R32G32B32A32_SFLOAT;
        attachments[4].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[4].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[4].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[4].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[4].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[4].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[4].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // Depth attachment
        attachments[5].format         = VK_FORMAT_D32_SFLOAT;
        attachments[5].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[5].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[5].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[5].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[5].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[5].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[5].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        std::array<VkAttachmentReference, 5> color_references = {};
        color_references[0].attachment                        = 0;
        color_references[0].layout                            = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_references[1].attachment                        = 1;
        color_references[1].layout                            = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_references[2].attachment                        = 2;
        color_references[2].layout                            = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_references[3].attachment                        = 3;
        color_references[3].layout                            = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_references[4].attachment                        = 4;
        color_references[4].layout                            = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depth_reference = {};
        depth_reference.attachment            = 5;
        depth_reference.layout                = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass_description    = {};
        subpass_description.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass_description.colorAttachmentCount    = color_references.size();
        subpass_description.pColorAttachments       = color_references.data();
        subpass_description.pDepthStencilAttachment = &depth_reference;
        subpass_description.inputAttachmentCount    = 0;
        subpass_description.pInputAttachments       = nullptr;
        subpass_description.preserveAttachmentCount = 0;
        subpass_description.pPreserveAttachments    = nullptr;
        subpass_description.pResolveAttachments     = nullptr;

        // Subpass dependencies for layout transitions
        std::array<VkSubpassDependency, 2> dependencies = {};

        dependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass      = 0;
        dependencies[0].srcStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        dependencies[1].srcSubpass      = 0;
        dependencies[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        VkRenderPassCreateInfo render_pass_create_info = {};
        render_pass_create_info.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_create_info.attachmentCount        = static_cast<uint32_t>(attachments.size());
        render_pass_create_info.pAttachments           = attachments.data();
        render_pass_create_info.subpassCount           = 1;
        render_pass_create_info.pSubpasses             = &subpass_description;
        render_pass_create_info.dependencyCount        = static_cast<uint32_t>(dependencies.size());
        render_pass_create_info.pDependencies          = dependencies.data();

        VK_CHECK(vkCreateRenderPass(get_device().get_handle(), &render_pass_create_info, nullptr, &render_passes.direct_pass));
    }

    // Translucent shadow maps
    {
        std::array<VkAttachmentDescription, 2> attachments = {};
        // Color attachment
        attachments[0].format         = VK_FORMAT_R32G32B32A32_SFLOAT;
        attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout  = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        attachments[0].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // Depth attachment
        attachments[1].format         = VK_FORMAT_D32_SFLOAT;
        attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        std::array<VkAttachmentReference, 1> color_references = {};
        color_references[0].attachment                        = 0;
        color_references[0].layout                            = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depth_reference = {};
        depth_reference.attachment            = 1;
        depth_reference.layout                = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass_description    = {};
        subpass_description.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass_description.colorAttachmentCount    = color_references.size();
        subpass_description.pColorAttachments       = color_references.data();
        subpass_description.pDepthStencilAttachment = &depth_reference;
        subpass_description.inputAttachmentCount    = 0;
        subpass_description.pInputAttachments       = nullptr;
        subpass_description.preserveAttachmentCount = 0;
        subpass_description.pPreserveAttachments    = nullptr;
        subpass_description.pResolveAttachments     = nullptr;

        // Subpass dependencies for layout transitions
        std::array<VkSubpassDependency, 2> dependencies;

        dependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass      = 0;
        dependencies[0].srcStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        dependencies[1].srcSubpass      = 0;
        dependencies[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        VkRenderPassCreateInfo render_pass_create_info = {};
        render_pass_create_info.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_create_info.attachmentCount        = static_cast<uint32_t>(attachments.size());
        render_pass_create_info.pAttachments           = attachments.data();
        render_pass_create_info.subpassCount           = 1;
        render_pass_create_info.pSubpasses             = &subpass_description;
        render_pass_create_info.dependencyCount        = static_cast<uint32_t>(dependencies.size());
        render_pass_create_info.pDependencies          = dependencies.data();

        VK_CHECK(vkCreateRenderPass(get_device().get_handle(), &render_pass_create_info, nullptr, &render_passes.trans_sm));
    }

    // Deferred shading
    {
        std::array<VkAttachmentDescription, 2> attachments = {};
        // Color attachment #1 (diffuse)
        attachments[0].format         = VK_FORMAT_R8G8B8A8_UNORM;
        attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // Depth attachment
        attachments[1].format         = VK_FORMAT_D32_SFLOAT;
        attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        std::array<VkAttachmentReference, 1> color_references = {};
        color_references[0].attachment                        = 0;
        color_references[0].layout                            = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depth_reference = {};
        depth_reference.attachment            = 1;
        depth_reference.layout                = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass_description    = {};
        subpass_description.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass_description.colorAttachmentCount    = color_references.size();
        subpass_description.pColorAttachments       = color_references.data();
        subpass_description.pDepthStencilAttachment = &depth_reference;
        subpass_description.inputAttachmentCount    = 0;
        subpass_description.pInputAttachments       = nullptr;
        subpass_description.preserveAttachmentCount = 0;
        subpass_description.pPreserveAttachments    = nullptr;
        subpass_description.pResolveAttachments     = nullptr;

        // Subpass dependencies for layout transitions
        std::array<VkSubpassDependency, 2> dependencies;

        dependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass      = 0;
        dependencies[0].srcStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        dependencies[1].srcSubpass      = 0;
        dependencies[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        VkRenderPassCreateInfo render_pass_create_info = {};
        render_pass_create_info.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_create_info.attachmentCount        = static_cast<uint32_t>(attachments.size());
        render_pass_create_info.pAttachments           = attachments.data();
        render_pass_create_info.subpassCount           = 1;
        render_pass_create_info.pSubpasses             = &subpass_description;
        render_pass_create_info.dependencyCount        = static_cast<uint32_t>(dependencies.size());
        render_pass_create_info.pDependencies          = dependencies.data();

        VK_CHECK(vkCreateRenderPass(get_device().get_handle(), &render_pass_create_info, nullptr, &render_passes.deferred));
    }
}

void LinSSScatter::setup_custom_framebuffers()
{
    // FBO for reflective shadow maps
    {
        // Create image and image view
        FBO &fbo = fbos.shadow_map;

        fbo.images.clear();
        fbo.images.emplace_back(get_device(),
                                VkExtent3D{SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1},
                                VK_FORMAT_R32G32B32A32_SFLOAT,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VMA_MEMORY_USAGE_GPU_ONLY,
                                VK_SAMPLE_COUNT_1_BIT,
                                1);

        fbo.images.emplace_back(get_device(),
                                VkExtent3D{SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1},
                                VK_FORMAT_R32G32B32A32_SFLOAT,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VMA_MEMORY_USAGE_GPU_ONLY,
                                VK_SAMPLE_COUNT_1_BIT,
                                1);

        fbo.images.emplace_back(get_device(),
                                VkExtent3D{SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1},
                                VK_FORMAT_R32G32B32A32_SFLOAT,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VMA_MEMORY_USAGE_GPU_ONLY,
                                VK_SAMPLE_COUNT_1_BIT,
                                1);

        fbo.images.emplace_back(get_device(),
                                VkExtent3D{SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1},
                                VK_FORMAT_D32_SFLOAT,
                                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VMA_MEMORY_USAGE_GPU_ONLY,
                                VK_SAMPLE_COUNT_1_BIT,
                                1);

        std::vector<VkImageView> attachments;
        fbo.views.clear();
        for (auto &image : fbo.images)
        {
            vkb::core::ImageView view{image, VK_IMAGE_VIEW_TYPE_2D, image.get_format()};
            attachments.push_back(view.get_handle());
            fbo.views.push_back(std::move(view));
        }

        VkFramebufferCreateInfo framebuffer_create_info = {};
        framebuffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebuffer_create_info.pNext                   = nullptr;
        framebuffer_create_info.renderPass              = render_passes.light_pass;
        framebuffer_create_info.attachmentCount         = attachments.size();
        framebuffer_create_info.pAttachments            = attachments.data();
        framebuffer_create_info.width                   = SHADOW_MAP_SIZE;
        framebuffer_create_info.height                  = SHADOW_MAP_SIZE;
        framebuffer_create_info.layers                  = 1;
        VK_CHECK(vkCreateFramebuffer(get_device().get_handle(), &framebuffer_create_info, nullptr, &fbo.fb));

        // Create sampler
        VkSamplerCreateInfo sampler = vkb::initializers::sampler_create_info();
        sampler.magFilter           = VK_FILTER_LINEAR;
        sampler.minFilter           = VK_FILTER_LINEAR;
        sampler.mipmapMode          = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler.addressModeU        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.addressModeV        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.addressModeW        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.mipLodBias          = 0.0f;
        sampler.compareOp           = VK_COMPARE_OP_NEVER;
        sampler.minLod              = 0.0f;
        sampler.maxLod              = 1.0f;

        if (get_device().get_gpu().get_features().samplerAnisotropy)
        {
            // Use max. level of anisotropy for this example
            sampler.maxAnisotropy    = get_device().get_gpu().get_properties().limits.maxSamplerAnisotropy;
            sampler.anisotropyEnable = VK_TRUE;
        }
        else
        {
            // The device does not support anisotropic filtering
            sampler.maxAnisotropy    = 1.0;
            sampler.anisotropyEnable = VK_FALSE;
        }
        sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        VK_CHECK(vkCreateSampler(get_device().get_handle(), &sampler, nullptr, &fbo.sampler));
    }

    // FBO for direct illumination
    {
        FBO &fbo = fbos.direct_pass;

        fbo.images.clear();

        fbo.images.emplace_back(get_device(),
                                VkExtent3D{get_render_context().get_surface_extent().width, get_render_context().get_surface_extent().height, 1},
                                VK_FORMAT_R32G32B32A32_SFLOAT,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                VMA_MEMORY_USAGE_GPU_ONLY,
                                VK_SAMPLE_COUNT_1_BIT,
                                1);

        fbo.images.emplace_back(get_device(),
                                VkExtent3D{get_render_context().get_surface_extent().width, get_render_context().get_surface_extent().height, 1},
                                VK_FORMAT_R32G32B32A32_SFLOAT,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VMA_MEMORY_USAGE_GPU_ONLY,
                                VK_SAMPLE_COUNT_1_BIT,
                                1);

        fbo.images.emplace_back(get_device(),
                                VkExtent3D{get_render_context().get_surface_extent().width, get_render_context().get_surface_extent().height, 1},
                                VK_FORMAT_R32G32B32A32_SFLOAT,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VMA_MEMORY_USAGE_GPU_ONLY,
                                VK_SAMPLE_COUNT_1_BIT,
                                1);

        fbo.images.emplace_back(get_device(),
                                VkExtent3D{get_render_context().get_surface_extent().width, get_render_context().get_surface_extent().height, 1},
                                VK_FORMAT_R32G32B32A32_SFLOAT,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VMA_MEMORY_USAGE_GPU_ONLY,
                                VK_SAMPLE_COUNT_1_BIT,
                                1);

        fbo.images.emplace_back(get_device(),
                                VkExtent3D{get_render_context().get_surface_extent().width, get_render_context().get_surface_extent().height, 1},
                                VK_FORMAT_R32G32B32A32_SFLOAT,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VMA_MEMORY_USAGE_GPU_ONLY,
                                VK_SAMPLE_COUNT_1_BIT,
                                1);

        fbo.images.emplace_back(get_device(),
                                VkExtent3D{get_render_context().get_surface_extent().width, get_render_context().get_surface_extent().height, 1},
                                VK_FORMAT_D32_SFLOAT,
                                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                VMA_MEMORY_USAGE_GPU_ONLY,
                                VK_SAMPLE_COUNT_1_BIT,
                                1);

        std::vector<VkImageView> attachments;
        fbo.views.clear();
        for (auto &image : fbo.images)
        {
            vkb::core::ImageView view{image, VK_IMAGE_VIEW_TYPE_2D, image.get_format()};
            attachments.push_back(view.get_handle());
            fbo.views.push_back(std::move(view));
        }

        VkFramebufferCreateInfo framebuffer_create_info = {};
        framebuffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebuffer_create_info.pNext                   = nullptr;
        framebuffer_create_info.renderPass              = render_passes.direct_pass;
        framebuffer_create_info.attachmentCount         = static_cast<uint32_t>(attachments.size());
        framebuffer_create_info.pAttachments            = attachments.data();
        framebuffer_create_info.width                   = get_render_context().get_surface_extent().width;
        framebuffer_create_info.height                  = get_render_context().get_surface_extent().height;
        framebuffer_create_info.layers                  = 1;
        VK_CHECK(vkCreateFramebuffer(get_device().get_handle(), &framebuffer_create_info, nullptr, &fbo.fb));

        // Create sampler
        VkSamplerCreateInfo sampler = vkb::initializers::sampler_create_info();
        sampler.magFilter           = VK_FILTER_LINEAR;
        sampler.minFilter           = VK_FILTER_LINEAR;
        sampler.mipmapMode          = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler.addressModeU        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.addressModeV        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.addressModeW        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.mipLodBias          = 0.0f;
        sampler.compareOp           = VK_COMPARE_OP_NEVER;
        sampler.minLod              = 0.0f;
        sampler.maxLod              = 1.0f;

        if (get_device().get_gpu().get_features().samplerAnisotropy)
        {
            // Use max. level of anisotropy for this example
            sampler.maxAnisotropy    = get_device().get_gpu().get_properties().limits.maxSamplerAnisotropy;
            sampler.anisotropyEnable = VK_TRUE;
        }
        else
        {
            // The device does not support anisotropic filtering
            sampler.maxAnisotropy    = 1.0;
            sampler.anisotropyEnable = VK_FALSE;
        }
        sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        VK_CHECK(vkCreateSampler(get_device().get_handle(), &sampler, nullptr, &fbo.sampler));
    }

    // FBO for Gaussian filter buffer
    {
        // Since this is just a buffer for Gaussian filtering,
        // there is no need to create framebuffer and sampler.
        FBO &fbo = fbos.gauss_filter_buffer;

        fbo.images.clear();

        const uint32_t mip_levels = max_mip_levels_surface();
        fbo.images.emplace_back(get_device(),
                                VkExtent3D{get_render_context().get_surface_extent().width, get_render_context().get_surface_extent().height, 1},
                                VK_FORMAT_R32G32B32A32_SFLOAT,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                VMA_MEMORY_USAGE_GPU_ONLY,
                                VK_SAMPLE_COUNT_1_BIT,
                                mip_levels);

        fbo.images.emplace_back(get_device(),
                                VkExtent3D{get_render_context().get_surface_extent().width, get_render_context().get_surface_extent().height, 1},
                                VK_FORMAT_R32G32B32A32_SFLOAT,
                                VK_IMAGE_USAGE_STORAGE_BIT,
                                VMA_MEMORY_USAGE_GPU_ONLY,
                                VK_SAMPLE_COUNT_1_BIT,
                                mip_levels);

        std::vector<VkImageView> attachments;
        fbo.views.clear();
        for (auto &image : fbo.images)
        {
            vkb::core::ImageView view{image, VK_IMAGE_VIEW_TYPE_2D, image.get_format()};
            attachments.push_back(view.get_handle());
            fbo.views.push_back(std::move(view));
        }

        // Create MIP levels for filtered image
        in_image_mip_level_views.resize(mip_levels);
        out_image_mip_level_views.resize(mip_levels);
        buf_image_mip_level_views.resize(mip_levels);

        // Create image with MIP levels
        VkImageCreateInfo image_create_info = vkb::initializers::image_create_info();
        image_create_info.imageType         = VK_IMAGE_TYPE_2D;
        image_create_info.format            = VK_FORMAT_R32G32B32A32_SFLOAT;
        image_create_info.mipLevels         = mip_levels;
        image_create_info.arrayLayers       = 1;
        image_create_info.samples           = VK_SAMPLE_COUNT_1_BIT;
        image_create_info.tiling            = VK_IMAGE_TILING_OPTIMAL;
        image_create_info.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
        image_create_info.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        image_create_info.extent            = VkExtent3D{get_render_context().get_surface_extent().width, get_render_context().get_surface_extent().height, 1};
        image_create_info.usage             = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        VK_CHECK(vkCreateImage(get_device().get_handle(), &image_create_info, nullptr, &G_ast_Phi_texture.image));

        VkMemoryRequirements memory_requirements = {};
        vkGetImageMemoryRequirements(get_device().get_handle(), G_ast_Phi_texture.image, &memory_requirements);

        VkMemoryAllocateInfo memory_allocate_info = vkb::initializers::memory_allocate_info();
        memory_allocate_info.allocationSize       = memory_requirements.size;
        memory_allocate_info.memoryTypeIndex      = get_device().get_memory_type(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(get_device().get_handle(), &memory_allocate_info, nullptr, &G_ast_Phi_texture.device_memory));
        VK_CHECK(vkBindImageMemory(get_device().get_handle(), G_ast_Phi_texture.image, G_ast_Phi_texture.device_memory, 0));

        for (uint32_t i = 0; i < mip_levels; i++)
        {
            // Create image view for a MIP level
            VkImageViewCreateInfo view_create_info           = vkb::initializers::image_view_create_info();
            view_create_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
            view_create_info.format                          = VK_FORMAT_R32G32B32A32_SFLOAT;
            view_create_info.components                      = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
            view_create_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            view_create_info.subresourceRange.baseMipLevel   = i;
            view_create_info.subresourceRange.levelCount     = 1;
            view_create_info.subresourceRange.baseArrayLayer = 0;
            view_create_info.subresourceRange.layerCount     = 1;

            view_create_info.image = G_ast_Phi_texture.image;
            VK_CHECK(vkCreateImageView(get_device().get_handle(), &view_create_info, nullptr, &out_image_mip_level_views[i]));

            view_create_info.image = fbos.gauss_filter_buffer.images[0].get_handle();
            VK_CHECK(vkCreateImageView(get_device().get_handle(), &view_create_info, nullptr, &in_image_mip_level_views[i]));

            view_create_info.image = fbos.gauss_filter_buffer.images[1].get_handle();
            VK_CHECK(vkCreateImageView(get_device().get_handle(), &view_create_info, nullptr, &buf_image_mip_level_views[i]));
        }

        // Create image view for G * Phi texture
        VkImageViewCreateInfo view           = vkb::initializers::image_view_create_info();
        view.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        view.format                          = VK_FORMAT_R32G32B32A32_SFLOAT;
        view.components                      = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
        view.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.baseMipLevel   = 0;
        view.subresourceRange.baseArrayLayer = 0;
        view.subresourceRange.layerCount     = 1;
        view.subresourceRange.levelCount     = mip_levels;
        view.image                           = G_ast_Phi_texture.image;
        VK_CHECK(vkCreateImageView(get_device().get_handle(), &view, nullptr, &G_ast_Phi_texture.view));

        // Create sampler for G * Phi texture
        VkSamplerCreateInfo sampler = vkb::initializers::sampler_create_info();
        sampler.magFilter           = VK_FILTER_LINEAR;
        sampler.minFilter           = VK_FILTER_LINEAR;
        sampler.mipmapMode          = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler.addressModeU        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeV        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeW        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.mipLodBias          = 0.0f;
        sampler.compareOp           = VK_COMPARE_OP_NEVER;
        sampler.minLod              = 0.0f;
        sampler.maxLod              = (float) mip_levels;

        if (get_device().get_gpu().get_features().samplerAnisotropy)
        {
            // Use max. level of anisotropy for this example
            sampler.maxAnisotropy    = get_device().get_gpu().get_properties().limits.maxSamplerAnisotropy;
            sampler.anisotropyEnable = VK_TRUE;
        }
        else
        {
            // The device does not support anisotropic filtering
            sampler.maxAnisotropy    = 1.0;
            sampler.anisotropyEnable = VK_FALSE;
        }
        sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        VK_CHECK(vkCreateSampler(get_device().get_handle(), &sampler, nullptr, &G_ast_Phi_texture.sampler));
    }

    // FBO for LinSSS accumulation
    {
        FBO &fbo = fbos.linsss;

        fbo.images.clear();

        const uint32_t mip_levels = std::ceil(std::log2(std::max(width, height)));
        fbo.images.emplace_back(get_device(),
                                VkExtent3D{get_render_context().get_surface_extent().width, get_render_context().get_surface_extent().height, 1},
                                VK_FORMAT_R32G32B32A32_SFLOAT,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VMA_MEMORY_USAGE_GPU_ONLY,
                                VK_SAMPLE_COUNT_1_BIT,
                                1);

        std::vector<VkImageView> attachments;
        fbo.views.clear();
        for (auto &image : fbo.images)
        {
            vkb::core::ImageView view{image, VK_IMAGE_VIEW_TYPE_2D, image.get_format()};
            attachments.push_back(view.get_handle());
            fbo.views.push_back(std::move(view));
        }

        // Create sampler
        VkSamplerCreateInfo sampler = vkb::initializers::sampler_create_info();
        sampler.magFilter           = VK_FILTER_LINEAR;
        sampler.minFilter           = VK_FILTER_LINEAR;
        sampler.mipmapMode          = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.addressModeV        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.addressModeW        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.mipLodBias          = 0.0f;
        sampler.compareOp           = VK_COMPARE_OP_NEVER;
        sampler.minLod              = 0.0f;
        sampler.maxLod              = 1.0f;

        if (get_device().get_gpu().get_features().samplerAnisotropy)
        {
            // Use max. level of anisotropy for this example
            sampler.maxAnisotropy    = get_device().get_gpu().get_properties().limits.maxSamplerAnisotropy;
            sampler.anisotropyEnable = VK_TRUE;
        }
        else
        {
            // The device does not support anisotropic filtering
            sampler.maxAnisotropy    = 1.0;
            sampler.anisotropyEnable = VK_FALSE;
        }
        sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        VK_CHECK(vkCreateSampler(get_device().get_handle(), &sampler, nullptr, &fbo.sampler));
    }

    // Transcluent shadow maps
    {
        const uint32_t tsm_width  = get_render_context().get_surface_extent().width / TSM_UPSAMPLE_RATIO;
        const uint32_t tsm_height = get_render_context().get_surface_extent().height / TSM_UPSAMPLE_RATIO;
        // Ping
        {
            FBO &fbo = fbos.trans_sm[0];

            fbo.images.clear();
            fbo.images.emplace_back(get_device(),
                                    VkExtent3D{tsm_width, tsm_height, 1},
                                    VK_FORMAT_R32G32B32A32_SFLOAT,
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                    VMA_MEMORY_USAGE_GPU_ONLY,
                                    VK_SAMPLE_COUNT_1_BIT,
                                    1);

            fbo.images.emplace_back(get_device(),
                                    VkExtent3D{tsm_width, tsm_height, 1},
                                    VK_FORMAT_D32_SFLOAT,
                                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                    VMA_MEMORY_USAGE_GPU_ONLY,
                                    VK_SAMPLE_COUNT_1_BIT,
                                    1);

            std::vector<VkImageView> attachments;
            fbo.views.clear();
            for (auto &image : fbo.images)
            {
                vkb::core::ImageView view{image, VK_IMAGE_VIEW_TYPE_2D, image.get_format()};
                attachments.push_back(view.get_handle());
                fbo.views.push_back(std::move(view));
            }

            VkFramebufferCreateInfo framebuffer_create_info = {};
            framebuffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebuffer_create_info.pNext                   = nullptr;
            framebuffer_create_info.renderPass              = render_passes.trans_sm;
            framebuffer_create_info.attachmentCount         = attachments.size();
            framebuffer_create_info.pAttachments            = attachments.data();
            framebuffer_create_info.width                   = tsm_width;
            framebuffer_create_info.height                  = tsm_height;
            framebuffer_create_info.layers                  = 1;
            VK_CHECK(vkCreateFramebuffer(get_device().get_handle(), &framebuffer_create_info, nullptr, &fbo.fb));

            // Create sampler
            VkSamplerCreateInfo sampler = vkb::initializers::sampler_create_info();
            sampler.magFilter           = VK_FILTER_LINEAR;
            sampler.minFilter           = VK_FILTER_LINEAR;
            sampler.mipmapMode          = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sampler.addressModeU        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sampler.addressModeV        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sampler.addressModeW        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sampler.mipLodBias          = 0.0f;
            sampler.compareOp           = VK_COMPARE_OP_NEVER;
            sampler.minLod              = 0.0f;
            sampler.maxLod              = 1.0f;

            if (get_device().get_gpu().get_features().samplerAnisotropy)
            {
                // Use max. level of anisotropy for this example
                sampler.maxAnisotropy    = get_device().get_gpu().get_properties().limits.maxSamplerAnisotropy;
                sampler.anisotropyEnable = VK_TRUE;
            }
            else
            {
                // The device does not support anisotropic filtering
                sampler.maxAnisotropy    = 1.0;
                sampler.anisotropyEnable = VK_FALSE;
            }
            sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
            VK_CHECK(vkCreateSampler(get_device().get_handle(), &sampler, nullptr, &fbo.sampler));

            // Clear image
            VkCommandBuffer command_buffer = get_device().create_command_buffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
            vkb::insert_image_memory_barrier(
                command_buffer,
                fbo.images[0].get_handle(),
                0,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

            VkClearColorValue clear_color{{0.0f, 0.0f, 0.0f, 1.0f}};
            VkImageSubresourceRange subresource_range;
            subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            subresource_range.baseMipLevel = 0;
            subresource_range.levelCount = 1;
            subresource_range.baseArrayLayer = 0;
            subresource_range.layerCount = 1;

            vkCmdClearColorImage(
                command_buffer,
                fbo.images[0].get_handle(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                &clear_color,
                1,
                &subresource_range);

            get_device().flush_command_buffer(command_buffer, queue, true);
        }

        // Pong
        {
            FBO &fbo = fbos.trans_sm[1];

            fbo.images.clear();
            fbo.images.emplace_back(get_device(),
                                    VkExtent3D{tsm_width, tsm_height, 1},
                                    VK_FORMAT_R32G32B32A32_SFLOAT,
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                    VMA_MEMORY_USAGE_GPU_ONLY,
                                    VK_SAMPLE_COUNT_1_BIT,
                                    1);

            fbo.images.emplace_back(get_device(),
                                    VkExtent3D{tsm_width, tsm_height, 1},
                                    VK_FORMAT_D32_SFLOAT,
                                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                    VMA_MEMORY_USAGE_GPU_ONLY,
                                    VK_SAMPLE_COUNT_1_BIT,
                                    1);

            std::vector<VkImageView> attachments;
            fbo.views.clear();
            for (auto &image : fbo.images)
            {
                vkb::core::ImageView view{image, VK_IMAGE_VIEW_TYPE_2D, image.get_format()};
                attachments.push_back(view.get_handle());
                fbo.views.push_back(std::move(view));
            }

            VkFramebufferCreateInfo framebuffer_create_info = {};
            framebuffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebuffer_create_info.pNext                   = nullptr;
            framebuffer_create_info.renderPass              = render_passes.trans_sm;
            framebuffer_create_info.attachmentCount         = attachments.size();
            framebuffer_create_info.pAttachments            = attachments.data();
            framebuffer_create_info.width                   = tsm_width;
            framebuffer_create_info.height                  = tsm_height;
            framebuffer_create_info.layers                  = 1;
            VK_CHECK(vkCreateFramebuffer(get_device().get_handle(), &framebuffer_create_info, nullptr, &fbo.fb));

            // Create sampler
            VkSamplerCreateInfo sampler = vkb::initializers::sampler_create_info();
            sampler.magFilter           = VK_FILTER_LINEAR;
            sampler.minFilter           = VK_FILTER_LINEAR;
            sampler.mipmapMode          = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sampler.addressModeU        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sampler.addressModeV        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sampler.addressModeW        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sampler.mipLodBias          = 0.0f;
            sampler.compareOp           = VK_COMPARE_OP_NEVER;
            sampler.minLod              = 0.0f;
            sampler.maxLod              = 1.0f;

            if (get_device().get_gpu().get_features().samplerAnisotropy)
            {
                // Use max. level of anisotropy for this example
                sampler.maxAnisotropy    = get_device().get_gpu().get_properties().limits.maxSamplerAnisotropy;
                sampler.anisotropyEnable = VK_TRUE;
            }
            else
            {
                // The device does not support anisotropic filtering
                sampler.maxAnisotropy    = 1.0;
                sampler.anisotropyEnable = VK_FALSE;
            }
            sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
            VK_CHECK(vkCreateSampler(get_device().get_handle(), &sampler, nullptr, &fbo.sampler));

            // Clear image
            VkCommandBuffer command_buffer = get_device().create_command_buffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
            vkb::insert_image_memory_barrier(
                command_buffer,
                fbo.images[0].get_handle(),
                0,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

            VkClearColorValue clear_color{{0.0f, 0.0f, 0.0f, 1.0f}};
            VkImageSubresourceRange subresource_range;
            subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            subresource_range.baseMipLevel = 0;
            subresource_range.levelCount = 1;
            subresource_range.baseArrayLayer = 0;
            subresource_range.layerCount = 1;

            vkCmdClearColorImage(
                command_buffer,
                fbo.images[0].get_handle(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                &clear_color,
                1,
                &subresource_range);

            get_device().flush_command_buffer(command_buffer, queue, true);
        }

        // Texture
        {
            // Create image with MIP levels
            VkImageCreateInfo image_create_info = vkb::initializers::image_create_info();
            image_create_info.imageType         = VK_IMAGE_TYPE_2D;
            image_create_info.format            = VK_FORMAT_R32G32B32A32_SFLOAT;
            image_create_info.mipLevels         = 1;
            image_create_info.arrayLayers       = 1;
            image_create_info.samples           = VK_SAMPLE_COUNT_1_BIT;
            image_create_info.tiling            = VK_IMAGE_TILING_OPTIMAL;
            image_create_info.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
            image_create_info.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
            image_create_info.extent            = VkExtent3D{tsm_width, tsm_height, 1};
            image_create_info.usage             = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            VK_CHECK(vkCreateImage(get_device().get_handle(), &image_create_info, nullptr, &tsm_texture.image));

            VkMemoryRequirements memory_requirements = {};
            vkGetImageMemoryRequirements(get_device().get_handle(), tsm_texture.image, &memory_requirements);

            VkMemoryAllocateInfo memory_allocate_info = vkb::initializers::memory_allocate_info();
            memory_allocate_info.allocationSize       = memory_requirements.size;
            memory_allocate_info.memoryTypeIndex      = get_device().get_memory_type(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VK_CHECK(vkAllocateMemory(get_device().get_handle(), &memory_allocate_info, nullptr, &tsm_texture.device_memory));
            VK_CHECK(vkBindImageMemory(get_device().get_handle(), tsm_texture.image, tsm_texture.device_memory, 0));

            // Create image view for G * Phi texture
            VkImageViewCreateInfo view           = vkb::initializers::image_view_create_info();
            view.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
            view.format                          = VK_FORMAT_R32G32B32A32_SFLOAT;
            view.components                      = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
            view.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            view.subresourceRange.baseMipLevel   = 0;
            view.subresourceRange.baseArrayLayer = 0;
            view.subresourceRange.layerCount     = 1;
            view.subresourceRange.levelCount     = 1;
            view.image                           = tsm_texture.image;
            VK_CHECK(vkCreateImageView(get_device().get_handle(), &view, nullptr, &tsm_texture.view));

            // Create sampler for G * Phi texture
            VkSamplerCreateInfo sampler = vkb::initializers::sampler_create_info();
            sampler.magFilter           = VK_FILTER_LINEAR;
            sampler.minFilter           = VK_FILTER_LINEAR;
            sampler.mipmapMode          = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sampler.addressModeU        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler.addressModeV        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler.addressModeW        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler.mipLodBias          = 0.0f;
            sampler.compareOp           = VK_COMPARE_OP_NEVER;
            sampler.minLod              = 0.0f;
            sampler.maxLod              = 1.0f;

            if (get_device().get_gpu().get_features().samplerAnisotropy)
            {
                // Use max. level of anisotropy for this example
                sampler.maxAnisotropy    = get_device().get_gpu().get_properties().limits.maxSamplerAnisotropy;
                sampler.anisotropyEnable = VK_TRUE;
            }
            else
            {
                // The device does not support anisotropic filtering
                sampler.maxAnisotropy    = 1.0;
                sampler.anisotropyEnable = VK_FALSE;
            }
            sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
            VK_CHECK(vkCreateSampler(get_device().get_handle(), &sampler, nullptr, &tsm_texture.sampler));
        }
    }

    // FBO for deferred shading
    {
        FBO &fbo = fbos.deferred;

        fbo.images.clear();
        fbo.images.emplace_back(get_device(),
                                VkExtent3D{get_render_context().get_surface_extent().width, get_render_context().get_surface_extent().height, 1},
                                VK_FORMAT_R8G8B8A8_UNORM,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VMA_MEMORY_USAGE_GPU_ONLY,
                                VK_SAMPLE_COUNT_1_BIT,
                                1);

        fbo.images.emplace_back(get_device(),
                                VkExtent3D{get_render_context().get_surface_extent().width, get_render_context().get_surface_extent().height, 1},
                                VK_FORMAT_D32_SFLOAT,
                                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                VMA_MEMORY_USAGE_GPU_ONLY,
                                VK_SAMPLE_COUNT_1_BIT,
                                1);

        std::vector<VkImageView> attachments;
        fbo.views.clear();
        for (auto &image : fbo.images)
        {
            vkb::core::ImageView view{image, VK_IMAGE_VIEW_TYPE_2D, image.get_format()};
            attachments.push_back(view.get_handle());
            fbo.views.push_back(std::move(view));
        }

        VkFramebufferCreateInfo framebuffer_create_info = {};
        framebuffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebuffer_create_info.pNext                   = nullptr;
        framebuffer_create_info.renderPass              = render_passes.deferred;
        framebuffer_create_info.attachmentCount         = attachments.size();
        framebuffer_create_info.pAttachments            = attachments.data();
        framebuffer_create_info.width                   = get_render_context().get_surface_extent().width;
        framebuffer_create_info.height                  = get_render_context().get_surface_extent().height;
        framebuffer_create_info.layers                  = 1;
        VK_CHECK(vkCreateFramebuffer(get_device().get_handle(), &framebuffer_create_info, nullptr, &fbo.fb));

        // Create sampler
        VkSamplerCreateInfo sampler = vkb::initializers::sampler_create_info();
        sampler.magFilter           = VK_FILTER_LINEAR;
        sampler.minFilter           = VK_FILTER_LINEAR;
        sampler.mipmapMode          = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.addressModeV        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.addressModeW        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler.mipLodBias          = 0.0f;
        sampler.compareOp           = VK_COMPARE_OP_NEVER;
        sampler.minLod              = 0.0f;
        sampler.maxLod              = 1.0f;

        if (get_device().get_gpu().get_features().samplerAnisotropy)
        {
            // Use max. level of anisotropy for this example
            sampler.maxAnisotropy    = get_device().get_gpu().get_properties().limits.maxSamplerAnisotropy;
            sampler.anisotropyEnable = VK_TRUE;
        }
        else
        {
            // The device does not support anisotropic filtering
            sampler.maxAnisotropy    = 1.0;
            sampler.anisotropyEnable = VK_FALSE;
        }
        sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        VK_CHECK(vkCreateSampler(get_device().get_handle(), &sampler, nullptr, &fbo.sampler));
    }
}

// Enable physical device features required for this example
void LinSSScatter::request_gpu_features(vkb::PhysicalDevice &gpu)
{
    // Enable anisotropic filtering if supported
    if (gpu.get_features().samplerAnisotropy)
    {
        gpu.get_mutable_requested_features().samplerAnisotropy = VK_TRUE;
    }
}

// Load envmap texture
void LinSSScatter::prepare_texture(LinSSScatter::Texture &texture, const std::string &filename, bool generateMipMap, float scale)
{
    // Split file extention
    std::string extension;
    {
        const int pos = filename.find_last_of('.');
        if (pos != std::string::npos)
        {
            extension = filename.substr(pos);
        }
    }

    // Load image data
    int                        image_width, image_height, image_channels = 4;
    VkFormat                   image_format;
    std::unique_ptr<uint8_t[]> image_data;
    if (extension == ".hdr")
    {
        float *bytes = stbi_loadf(filename.c_str(), &image_width, &image_height, nullptr, STBI_rgb_alpha);
        if (!bytes)
        {
            LOGE("Failed to load image file: {}", filename);
        }

        if (scale != 1.0f)
        {
            for (uint32_t i = 0; i < image_width * image_height * image_channels; i++)
            {
                bytes[i] *= scale;
            }
        }

        image_format = VK_FORMAT_R32G32B32A32_SFLOAT;
        image_data   = std::make_unique<uint8_t[]>(image_width * image_height * image_channels * sizeof(float));
        std::memcpy(image_data.get(), bytes, image_width * image_height * image_channels * sizeof(float));
        stbi_image_free(bytes);
    }
    else
    {
        uint8_t *bytes = stbi_load(filename.c_str(), &image_width, &image_height, nullptr, STBI_rgb_alpha);
        if (!bytes)
        {
            LOGE("Failed to load image file: {}", vkb::to_string(filename));
        }

        if (scale != 1.0f)
        {
            for (uint32_t i = 0; i < image_width * image_height * image_channels; i++)
            {
                const float v = static_cast<float>(bytes[i]) * scale;
                bytes[i] = static_cast<uint8_t>(std::max(0.0f, std::min(v, 255.0f)));
            }
        }

        image_format = VK_FORMAT_R8G8B8A8_UNORM;
        image_data   = std::make_unique<uint8_t[]>(image_width * image_height * image_channels * sizeof(uint8_t));
        std::memcpy(image_data.get(), bytes, image_width * image_height * image_channels * sizeof(uint8_t));
        stbi_image_free(bytes);
    }

    texture.width      = image_width;
    texture.height     = image_height;
    texture.mip_levels = generateMipMap ? std::ceil(std::log2(std::max(image_width, image_height))) : 1;

    // We prefer using staging to copy the texture data to a device local optimal image
    VkMemoryAllocateInfo memory_allocate_info = vkb::initializers::memory_allocate_info();
    VkMemoryRequirements memory_requirements  = {};

    // Copy data to an optimal tiled image
    // This loads the texture data into a host local buffer that is copied to the optimal tiled image on the device

    // Create a host-visible staging buffer that contains the raw image data
    // This buffer will be the data source for copying texture data to the optimal tiled image on the device
    VkBuffer       staging_buffer;
    VkDeviceMemory staging_memory;

    VkBufferCreateInfo buffer_create_info = vkb::initializers::buffer_create_info();
    buffer_create_info.size               = image_width * image_height * sizeof(float) * 4;
    // This buffer is used as a transfer source for the buffer copy
    buffer_create_info.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(get_device().get_handle(), &buffer_create_info, nullptr, &staging_buffer));

    // Get memory requirements for the staging buffer (alignment, memory type bits)
    vkGetBufferMemoryRequirements(get_device().get_handle(), staging_buffer, &memory_requirements);
    memory_allocate_info.allocationSize = memory_requirements.size;
    // Get memory type index for a host visible buffer
    memory_allocate_info.memoryTypeIndex = get_device().get_memory_type(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(get_device().get_handle(), &memory_allocate_info, nullptr, &staging_memory));
    VK_CHECK(vkBindBufferMemory(get_device().get_handle(), staging_buffer, staging_memory, 0));

    // Copy texture data into host local staging buffer
    uint32_t image_size = sizeof(float) * image_width * image_height * 4;

    uint8_t *data;
    VK_CHECK(vkMapMemory(get_device().get_handle(), staging_memory, 0, memory_requirements.size, 0, (void **) &data));
    std::memcpy(data, image_data.get(), image_size);
    vkUnmapMemory(get_device().get_handle(), staging_memory);

    // Setup buffer copy regions for each mip level
    std::vector<VkBufferImageCopy> buffer_copy_regions;
    for (uint32_t i = 0; i < texture.mip_levels; i++)
    {
        VkBufferImageCopy buffer_copy_region               = {};
        buffer_copy_region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        buffer_copy_region.imageSubresource.mipLevel       = i;
        buffer_copy_region.imageSubresource.baseArrayLayer = 0;
        buffer_copy_region.imageSubresource.layerCount     = 1;
        buffer_copy_region.imageExtent.width               = image_width >> i;
        buffer_copy_region.imageExtent.height              = image_height >> i;
        buffer_copy_region.imageExtent.depth               = 1;
        buffer_copy_region.bufferOffset                    = 0;
        buffer_copy_regions.push_back(buffer_copy_region);
    }

    // Create optimal tiled target image on the device
    VkImageCreateInfo image_create_info = vkb::initializers::image_create_info();
    image_create_info.imageType         = VK_IMAGE_TYPE_2D;
    image_create_info.format            = image_format;
    image_create_info.mipLevels         = texture.mip_levels;
    image_create_info.arrayLayers       = 1;
    image_create_info.samples           = VK_SAMPLE_COUNT_1_BIT;
    image_create_info.tiling            = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
    // Set initial layout of the image to undefined
    image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_create_info.extent        = VkExtent3D{texture.width, texture.height, 1};
    image_create_info.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VK_CHECK(vkCreateImage(get_device().get_handle(), &image_create_info, nullptr, &texture.image));

    vkGetImageMemoryRequirements(get_device().get_handle(), texture.image, &memory_requirements);
    memory_allocate_info.allocationSize  = memory_requirements.size;
    memory_allocate_info.memoryTypeIndex = get_device().get_memory_type(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(get_device().get_handle(), &memory_allocate_info, nullptr, &texture.device_memory));
    VK_CHECK(vkBindImageMemory(get_device().get_handle(), texture.image, texture.device_memory, 0));

    VkCommandBuffer copy_command = device->create_command_buffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    // Image memory barriers for the texture image

    // The sub resource range describes the regions of the image that will be transitioned using the memory barriers below
    VkImageSubresourceRange subresource_range = {};
    // Image only contains color data
    subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    // Start at first mip level
    subresource_range.baseMipLevel = 0;
    // We will transition on all mip levels
    subresource_range.levelCount = texture.mip_levels;
    // The 2D texture only has one layer
    subresource_range.layerCount = 1;

    // Transition the texture image layout to transfer target, so we can safely copy our buffer data to it.
    VkImageMemoryBarrier image_memory_barrier = vkb::initializers::image_memory_barrier();

    image_memory_barrier.image            = texture.image;
    image_memory_barrier.subresourceRange = subresource_range;
    image_memory_barrier.srcAccessMask    = 0;
    image_memory_barrier.dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT;
    image_memory_barrier.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
    image_memory_barrier.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    // Insert a memory dependency at the proper pipeline stages that will execute the image layout transition
    // Source pipeline stage is host write/read exection (VK_PIPELINE_STAGE_HOST_BIT)
    // Destination pipeline stage is copy command exection (VK_PIPELINE_STAGE_TRANSFER_BIT)
    vkCmdPipelineBarrier(
        copy_command,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &image_memory_barrier);

    // Copy mip levels from staging buffer
    vkCmdCopyBufferToImage(
        copy_command,
        staging_buffer,
        texture.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        static_cast<uint32_t>(buffer_copy_regions.size()),
        buffer_copy_regions.data());

    // Once the data has been uploaded we transfer to the texture image to the shader read layout, so it can be sampled from
    image_memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image_memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image_memory_barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_memory_barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Insert a memory dependency at the proper pipeline stages that will execute the image layout transition
    // Source pipeline stage stage is copy command exection (VK_PIPELINE_STAGE_TRANSFER_BIT)
    // Destination pipeline stage fragment shader access (VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
    vkCmdPipelineBarrier(
        copy_command,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &image_memory_barrier);

    // Store current layout for later reuse
    texture.image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    device->flush_command_buffer(copy_command, queue, true);

    // Clean up staging resources
    vkFreeMemory(get_device().get_handle(), staging_memory, nullptr);
    vkDestroyBuffer(get_device().get_handle(), staging_buffer, nullptr);

    // Create a texture sampler
    // In Vulkan textures are accessed by samplers
    // This separates all the sampling information from the texture data. This means you could have multiple sampler objects for the same texture with different settings
    // Note: Similar to the samplers available with OpenGL 3.3
    VkSamplerCreateInfo sampler = vkb::initializers::sampler_create_info();
    sampler.magFilter           = VK_FILTER_LINEAR;
    sampler.minFilter           = VK_FILTER_LINEAR;
    sampler.mipmapMode          = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.addressModeV        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.addressModeW        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.mipLodBias          = 0.0f;
    sampler.compareOp           = VK_COMPARE_OP_NEVER;
    sampler.minLod              = 0.0f;
    // Set max level-of-detail to mip level count of the texture
    sampler.maxLod = (float) texture.mip_levels;
    // Enable anisotropic filtering
    // This feature is optional, so we must check if it's supported on the device
    if (get_device().get_gpu().get_features().samplerAnisotropy)
    {
        // Use max. level of anisotropy for this example
        sampler.maxAnisotropy    = get_device().get_gpu().get_properties().limits.maxSamplerAnisotropy;
        sampler.anisotropyEnable = VK_TRUE;
    }
    else
    {
        // The device does not support anisotropic filtering
        sampler.maxAnisotropy    = 1.0;
        sampler.anisotropyEnable = VK_FALSE;
    }
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK(vkCreateSampler(get_device().get_handle(), &sampler, nullptr, &texture.sampler));

    // Create image view
    // Textures are not directly accessed by the shaders and
    // are abstracted by image views containing additional
    // information and sub resource ranges
    VkImageViewCreateInfo view = vkb::initializers::image_view_create_info();
    view.viewType              = VK_IMAGE_VIEW_TYPE_2D;
    view.format                = image_format;
    view.components            = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
    // The subresource range describes the set of mip levels (and array layers) that can be accessed through this image view
    // It's possible to create multiple image views for a single image referring to different (and/or overlapping) ranges of the image
    view.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.baseMipLevel   = 0;
    view.subresourceRange.baseArrayLayer = 0;
    view.subresourceRange.layerCount     = 1;
    // Linear tiling usually won't support mip maps
    // Only set mip map count if optimal tiling is used
    view.subresourceRange.levelCount = texture.mip_levels;
    // The view will be based on the texture's image
    view.image = texture.image;
    VK_CHECK(vkCreateImageView(get_device().get_handle(), &view, nullptr, &texture.view));
}

void LinSSScatter::generate_mipmap(VkCommandBuffer command_buffer, VkImage image, uint32_t image_width, uint32_t image_height, VkFormat format, uint32_t mip_levels)
{
    // Setup buffer copy regions for each mip level
    int32_t mipmap_width  = static_cast<int32_t>(image_width);
    int32_t mipmap_height = static_cast<int32_t>(image_height);

    // Change image layout of color attachment
    vkb::insert_image_memory_barrier(
        command_buffer,
        image,
        0,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    // Copy first MIP level
    {
        // Pre-copy barrier
        vkb::insert_image_memory_barrier(
            command_buffer,
            fbos.gauss_filter_buffer.images[0].get_handle(),
            0,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

        // Setup image copy
        VkImageCopy image_copy                   = {};
        image_copy.extent                        = VkExtent3D{image_width, image_height, 1};
        image_copy.srcOffset                     = {0, 0, 0};
        image_copy.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        image_copy.srcSubresource.mipLevel       = 0;
        image_copy.srcSubresource.baseArrayLayer = 0;
        image_copy.srcSubresource.layerCount     = 1;
        image_copy.dstOffset                     = {0, 0, 0};
        image_copy.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        image_copy.dstSubresource.mipLevel       = 0;
        image_copy.dstSubresource.baseArrayLayer = 0;
        image_copy.dstSubresource.layerCount     = 1;

        // Image blit
        vkCmdCopyImage(
            command_buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            fbos.gauss_filter_buffer.images[0].get_handle(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &image_copy);

        // Post-copy barrier
        vkb::insert_image_memory_barrier(
            command_buffer,
            fbos.gauss_filter_buffer.images[0].get_handle(),
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    }

    // Copy image by climbing MIP levels
    for (uint32_t i = 0; i < mip_levels - 1; i++)
    {
        // Setup image blit
        VkImageBlit image_blit                   = {};
        image_blit.srcOffsets[0]                 = {0, 0, 0};
        image_blit.srcOffsets[1]                 = {mipmap_width, mipmap_height, 1};
        image_blit.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        image_blit.srcSubresource.mipLevel       = i;
        image_blit.srcSubresource.baseArrayLayer = 0;
        image_blit.srcSubresource.layerCount     = 1;
        image_blit.dstOffsets[0]                 = {0, 0, 0};
        image_blit.dstOffsets[1]                 = {std::max(1, mipmap_width / 2), std::max(1, mipmap_height / 2), 1};
        image_blit.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        image_blit.dstSubresource.mipLevel       = i + 1;
        image_blit.dstSubresource.baseArrayLayer = 0;
        image_blit.dstSubresource.layerCount     = 1;

        if (mipmap_width > 1)
            mipmap_width /= 2;
        if (mipmap_height > 1)
            mipmap_height /= 2;

        vkb::insert_image_memory_barrier(
            command_buffer,
            fbos.gauss_filter_buffer.images[0].get_handle(),
            0,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            {VK_IMAGE_ASPECT_COLOR_BIT, i + 1, 1, 0, 1});

        vkCmdBlitImage(
            command_buffer,
            fbos.gauss_filter_buffer.images[0].get_handle(),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            fbos.gauss_filter_buffer.images[0].get_handle(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &image_blit,
            VK_FILTER_LINEAR);

        vkb::insert_image_memory_barrier(
            command_buffer,
            fbos.gauss_filter_buffer.images[0].get_handle(),
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            {VK_IMAGE_ASPECT_COLOR_BIT, i + 1, 1, 0, 1});
    }

    vkb::insert_image_memory_barrier(
        command_buffer,
        fbos.gauss_filter_buffer.images[0].get_handle(),
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, mip_levels, 0, 1});
}

void LinSSScatter::destroy_custom_render_passes()
{
    vkDestroyRenderPass(get_device().get_handle(), render_passes.light_pass, nullptr);
    vkDestroyRenderPass(get_device().get_handle(), render_passes.direct_pass, nullptr);
    vkDestroyRenderPass(get_device().get_handle(), render_passes.deferred, nullptr);
    vkDestroyRenderPass(get_device().get_handle(), render_passes.trans_sm, nullptr);
}

void LinSSScatter::destroy_custom_framebuffers()
{
    {
        FBO &fbo = fbos.shadow_map;
        vkDestroyFramebuffer(get_device().get_handle(), fbo.fb, nullptr);
        vkDestroySampler(get_device().get_handle(), fbo.sampler, nullptr);
    }

    {
        FBO &fbo = fbos.direct_pass;
        vkDestroyFramebuffer(get_device().get_handle(), fbo.fb, nullptr);
        vkDestroySampler(get_device().get_handle(), fbo.sampler, nullptr);
    }

    {
        // Nothing to do for buffer image
        FBO &fbo = fbos.gauss_filter_buffer;

        // Destroy image for filtered MIP map
        destroy_texture(G_ast_Phi_texture);
        const uint32_t mip_levels = max_mip_levels_surface();
        for (uint32_t i = 0; i < mip_levels; i++)
        {
            vkDestroyImageView(get_device().get_handle(), in_image_mip_level_views[i], nullptr);
            vkDestroyImageView(get_device().get_handle(), out_image_mip_level_views[i], nullptr);
            vkDestroyImageView(get_device().get_handle(), buf_image_mip_level_views[i], nullptr);
        }
    }

    {
        FBO &fbo = fbos.linsss;
        vkDestroySampler(get_device().get_handle(), fbo.sampler, nullptr);
    }

    {
        destroy_texture(tsm_texture);
        vkDestroyFramebuffer(get_device().get_handle(), fbos.trans_sm[0].fb, nullptr);
        vkDestroySampler(get_device().get_handle(), fbos.trans_sm[0].sampler, nullptr);
        vkDestroyFramebuffer(get_device().get_handle(), fbos.trans_sm[1].fb, nullptr);
        vkDestroySampler(get_device().get_handle(), fbos.trans_sm[1].sampler, nullptr);
    }

    {
        FBO &fbo = fbos.deferred;
        vkDestroyFramebuffer(get_device().get_handle(), fbo.fb, nullptr);
        vkDestroySampler(get_device().get_handle(), fbo.sampler, nullptr);
    }
}

void LinSSScatter::destroy_texture(LinSSScatter::Texture texture)
{
    vkDestroyImageView(get_device().get_handle(), texture.view, nullptr);
    vkDestroyImage(get_device().get_handle(), texture.image, nullptr);
    vkDestroySampler(get_device().get_handle(), texture.sampler, nullptr);
    vkFreeMemory(get_device().get_handle(), texture.device_memory, nullptr);
}

void LinSSScatter::destroy_bssrdf(LinSSScatter::BSSRDF bssrdf)
{
    vkDestroyImageView(get_device().get_handle(), bssrdf.view_W, nullptr);
    vkDestroyImageView(get_device().get_handle(), bssrdf.view_G_ast_W, nullptr);
    vkDestroyImage(get_device().get_handle(), bssrdf.image_W, nullptr);
    vkDestroyImage(get_device().get_handle(), bssrdf.image_G_ast_W, nullptr);
    vkDestroySampler(get_device().get_handle(), bssrdf.sampler, nullptr);
    vkFreeMemory(get_device().get_handle(), bssrdf.device_memory_W, nullptr);
    vkFreeMemory(get_device().get_handle(), bssrdf.device_memory_G_ast_W, nullptr);
}

void LinSSScatter::gauss_filter_to_mipmap_compute(VkCommandBuffer command_buffer, uint32_t image_width, uint32_t image_height, uint32_t mip_levels)
{
    // Copy first MIP level
    vkb::insert_image_memory_barrier(
        command_buffer,
        fbos.gauss_filter_buffer.images[0].get_handle(),
        VK_ACCESS_SHADER_READ_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    vkb::insert_image_memory_barrier(
        command_buffer,
        G_ast_Phi_texture.image,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    // Setup image copy
    VkImageCopy image_copy                   = {};
    image_copy.extent                        = VkExtent3D{image_width, image_height, 1};
    image_copy.srcOffset                     = {0, 0, 0};
    image_copy.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    image_copy.srcSubresource.mipLevel       = 0;
    image_copy.srcSubresource.baseArrayLayer = 0;
    image_copy.srcSubresource.layerCount     = 1;
    image_copy.dstOffset                     = {0, 0, 0};
    image_copy.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    image_copy.dstSubresource.mipLevel       = 0;
    image_copy.dstSubresource.baseArrayLayer = 0;
    image_copy.dstSubresource.layerCount     = 1;

    vkCmdCopyImage(
        command_buffer,
        fbos.gauss_filter_buffer.images[0].get_handle(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        G_ast_Phi_texture.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &image_copy);

    vkb::insert_image_memory_barrier(
        command_buffer,
        G_ast_Phi_texture.image,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    // Skip a filter for the base MIP level.
    // It's just a copy for incident irradiance map.
    uint32_t mipmap_width  = image_width;
    uint32_t mipmap_height = image_height;
    for (uint32_t i = 1; i < mip_levels; i++)
    {
        if (mipmap_width > 1)
            mipmap_width /= 2;
        if (mipmap_height > 1)
            mipmap_height /= 2;

        // Change image layout
        vkb::insert_image_memory_barrier(
            command_buffer,
            fbos.gauss_filter_buffer.images[1].get_handle(),
            0,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            {VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1});

        vkb::insert_image_memory_barrier(
            command_buffer,
            G_ast_Phi_texture.image,
            0,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            {VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1});

        // Dispatch
        const uint32_t local_size  = 32;
        const uint32_t num_group_x = (mipmap_width + local_size - 1) / local_size;
        const uint32_t num_group_y = (mipmap_height + local_size - 1) / local_size;
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.gauss_filter);

        // Horizontal filter
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layouts.gauss_filter, 0, 1, &descriptor_sets.gauss_horz_filter[i], 0, nullptr);
        vkCmdDispatch(command_buffer, num_group_x, num_group_y, 1);

        vkb::insert_image_memory_barrier(
            command_buffer,
            G_ast_Phi_texture.image,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            {VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1});

        // Vertical filter
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layouts.gauss_filter, 0, 1, &descriptor_sets.gauss_vert_filter[i], 0, nullptr);
        vkCmdDispatch(command_buffer, num_group_x, num_group_y, 1);

        vkb::insert_image_memory_barrier(
            command_buffer,
            G_ast_Phi_texture.image,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            {VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1});
    }
}

void LinSSScatter::linsss_accumulate_compute(VkCommandBuffer command_buffer)
{
    // Bind descriptor set
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layouts.linsss, 0, 1, &descriptor_sets.linsss, 0, nullptr);

    // Image memory barrier
    vkb::insert_image_memory_barrier(
        command_buffer,
        fbos.linsss.images[0].get_handle(),
        VK_ACCESS_SHADER_READ_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    // Dispatch
    const uint32_t local_size  = 32;
    const uint32_t num_group_x = (width + local_size - 1) / local_size;
    const uint32_t num_group_y = (height + local_size - 1) / local_size;
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.linsss);
    vkCmdDispatch(command_buffer, num_group_x, num_group_y, 1);

    vkb::insert_image_memory_barrier(
        command_buffer,
        fbos.linsss.images[0].get_handle(),
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
}

void LinSSScatter::build_command_buffers()
{
    // Update descriptor set
    update_descriptor_set();

    // Build command buffer
    VkCommandBufferBeginInfo command_buffer_begin_info = vkb::initializers::command_buffer_begin_info();

    VkClearValue light_pass_clear_values[4];
    light_pass_clear_values[0].color        = default_clear_color;
    light_pass_clear_values[1].color        = default_clear_color;
    light_pass_clear_values[2].color        = default_clear_color;
    light_pass_clear_values[3].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo render_light_pass_begin_info    = vkb::initializers::render_pass_begin_info();
    render_light_pass_begin_info.renderPass               = render_passes.light_pass;
    render_light_pass_begin_info.renderArea.offset.x      = 0;
    render_light_pass_begin_info.renderArea.offset.y      = 0;
    render_light_pass_begin_info.renderArea.extent.width  = SHADOW_MAP_SIZE;
    render_light_pass_begin_info.renderArea.extent.height = SHADOW_MAP_SIZE;
    render_light_pass_begin_info.clearValueCount          = 4;
    render_light_pass_begin_info.pClearValues             = light_pass_clear_values;

    VkClearValue direct_pass_clear_values[6];
    direct_pass_clear_values[0].color        = default_clear_color;
    direct_pass_clear_values[1].color        = default_clear_color;
    direct_pass_clear_values[2].color        = default_clear_color;
    direct_pass_clear_values[3].color        = default_clear_color;
    direct_pass_clear_values[4].color        = default_clear_color;
    direct_pass_clear_values[5].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo render_direct_pass_begin_info    = vkb::initializers::render_pass_begin_info();
    render_direct_pass_begin_info.renderPass               = render_passes.direct_pass;
    render_direct_pass_begin_info.renderArea.offset.x      = 0;
    render_direct_pass_begin_info.renderArea.offset.y      = 0;
    render_direct_pass_begin_info.renderArea.extent.width  = width;
    render_direct_pass_begin_info.renderArea.extent.height = height;
    render_direct_pass_begin_info.clearValueCount          = 6;
    render_direct_pass_begin_info.pClearValues             = direct_pass_clear_values;

    VkClearValue clear_values[2];
    clear_values[0].color        = default_clear_color;
    clear_values[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo render_tsm_pass_begin_info    = vkb::initializers::render_pass_begin_info();
    render_tsm_pass_begin_info.renderPass               = render_passes.trans_sm;
    render_tsm_pass_begin_info.renderArea.offset.x      = 0;
    render_tsm_pass_begin_info.renderArea.offset.y      = 0;
    render_tsm_pass_begin_info.renderArea.extent.width  = width / TSM_UPSAMPLE_RATIO;
    render_tsm_pass_begin_info.renderArea.extent.height = height / TSM_UPSAMPLE_RATIO;
    render_tsm_pass_begin_info.clearValueCount          = 2;
    render_tsm_pass_begin_info.pClearValues             = clear_values;

    VkRenderPassBeginInfo render_deferred_pass_begin_info    = vkb::initializers::render_pass_begin_info();
    render_deferred_pass_begin_info.renderPass               = render_passes.deferred;
    render_deferred_pass_begin_info.renderArea.offset.x      = 0;
    render_deferred_pass_begin_info.renderArea.offset.y      = 0;
    render_deferred_pass_begin_info.renderArea.extent.width  = width;
    render_deferred_pass_begin_info.renderArea.extent.height = height;
    render_deferred_pass_begin_info.clearValueCount          = 2;
    render_deferred_pass_begin_info.pClearValues             = clear_values;

    VkRenderPassBeginInfo render_postprocess_begin_info    = vkb::initializers::render_pass_begin_info();
    render_postprocess_begin_info.renderPass               = render_pass;
    render_postprocess_begin_info.renderArea.offset.x      = 0;
    render_postprocess_begin_info.renderArea.offset.y      = 0;
    render_postprocess_begin_info.renderArea.extent.width  = width;
    render_postprocess_begin_info.renderArea.extent.height = height;
    render_postprocess_begin_info.clearValueCount          = 2;
    render_postprocess_begin_info.pClearValues             = clear_values;

    VkDeviceSize offsets[1] = {0};

    for (int32_t i = 0; i < draw_cmd_buffers.size(); ++i)
    {
        // BEGIN
        VK_CHECK(vkBeginCommandBuffer(draw_cmd_buffers[i], &command_buffer_begin_info));
        {
            // Begin render pass (light pass)
            render_light_pass_begin_info.framebuffer = fbos.shadow_map.fb;
            vkCmdBeginRenderPass(draw_cmd_buffers[i], &render_light_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);
            {
                // Viewport
                VkViewport viewport = vkb::initializers::viewport((float) SHADOW_MAP_SIZE, (float) SHADOW_MAP_SIZE, 0.0f, 1.0f);
                vkCmdSetViewport(draw_cmd_buffers[i], 0, 1, &viewport);

                // Scissor
                VkRect2D scissor = vkb::initializers::rect2D(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 0, 0);
                vkCmdSetScissor(draw_cmd_buffers[i], 0, 1, &scissor);

                // Pipeline layout
                vkCmdBindDescriptorSets(draw_cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layouts.light_pass, 0, 1, &descriptor_sets.light_pass, 0, nullptr);

                // Draw
                if (ubo_fs.light_type == LightType::Point)
                {
                    vkCmdBindPipeline(draw_cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.light_pass);
                    vkCmdBindVertexBuffers(draw_cmd_buffers[i], 0, 1, model.vertex_buffer->get(), offsets);
                    vkCmdBindIndexBuffer(draw_cmd_buffers[i], model.index_buffer->get_handle(), 0, VK_INDEX_TYPE_UINT32);
                    vkCmdDrawIndexed(draw_cmd_buffers[i], model.index_count, 1, 0, 0, 0);
                }
            }
            // End render pass (light pass)
            vkCmdEndRenderPass(draw_cmd_buffers[i]);

            // Begin render pass (direct pass)
            render_direct_pass_begin_info.framebuffer = fbos.direct_pass.fb;
            vkCmdBeginRenderPass(draw_cmd_buffers[i], &render_direct_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);
            {
                // Viewport
                VkViewport viewport = vkb::initializers::viewport((float) width, (float) height, 0.0f, 1.0f);
                vkCmdSetViewport(draw_cmd_buffers[i], 0, 1, &viewport);

                // Scissor
                VkRect2D scissor = vkb::initializers::rect2D(width, height, 0, 0);
                vkCmdSetScissor(draw_cmd_buffers[i], 0, 1, &scissor);

                // Pipeline layout
                vkCmdBindDescriptorSets(draw_cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layouts.direct_pass, 0, 1, &descriptor_sets.direct_pass, 0, nullptr);

                // Draw
                vkCmdBindPipeline(draw_cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.direct_pass);
                vkCmdBindVertexBuffers(draw_cmd_buffers[i], 0, 1, model.vertex_buffer->get(), offsets);
                vkCmdBindIndexBuffer(draw_cmd_buffers[i], model.index_buffer->get_handle(), 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(draw_cmd_buffers[i], model.index_count, 1, 0, 0, 0);
            }
            // End render pass (direct pass)
            vkCmdEndRenderPass(draw_cmd_buffers[i]);

            // Generate MIP Map
            {
                vkb::core::Image &image        = fbos.direct_pass.images[0];
                const uint32_t    image_width  = image.get_extent().width;
                const uint32_t    image_height = image.get_extent().height;
                const uint32_t    mip_levels   = std::ceil(std::log2(std::max(image_width, image_height)));
                generate_mipmap(draw_cmd_buffers[i], image.get_handle(), image_width, image_height, image.get_format(), mip_levels);
            }

            // Change image layout
            vkb::insert_image_memory_barrier(
                draw_cmd_buffers[i],
                fbos.shadow_map.images[0].get_handle(),
                0,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

            vkb::insert_image_memory_barrier(
                draw_cmd_buffers[i],
                fbos.shadow_map.images[1].get_handle(),
                0,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

            vkb::insert_image_memory_barrier(
                draw_cmd_buffers[i],
                fbos.shadow_map.images[2].get_handle(),
                0,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

            vkb::insert_image_memory_barrier(
                draw_cmd_buffers[i],
                fbos.direct_pass.images[1].get_handle(),
                0,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

            vkb::insert_image_memory_barrier(
                draw_cmd_buffers[i],
                fbos.direct_pass.images[2].get_handle(),
                0,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

            vkb::insert_image_memory_barrier(
                draw_cmd_buffers[i],
                fbos.direct_pass.images[3].get_handle(),
                0,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

            vkb::insert_image_memory_barrier(
                draw_cmd_buffers[i],
                fbos.direct_pass.images[4].get_handle(),
                0,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

            // Compute pass (gauss filter)
            {
                vkb::core::Image &image        = fbos.direct_pass.images[0];
                const uint32_t    image_width  = image.get_extent().width;
                const uint32_t    image_height = image.get_extent().height;
                const uint32_t    mip_levels   = std::ceil(std::log2(std::max(image_width, image_height)));

                gauss_filter_to_mipmap_compute(draw_cmd_buffers[i], image_width, image_height, mip_levels);
            }

            // Compute pass (linsss accumulate)
            {
                linsss_accumulate_compute(draw_cmd_buffers[i]);
            }

            // Translucent shadow maps
            const int ping_index = i % 2;
            const int pong_index = 1 - ping_index;

            // When TSM is enabled
            if (enable_tsm)
            {
                vkb::insert_image_memory_barrier(
                    draw_cmd_buffers[i],
                    fbos.trans_sm[ping_index].images[0].get_handle(),
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

                vkb::insert_image_memory_barrier(
                    draw_cmd_buffers[i],
                    fbos.trans_sm[pong_index].images[0].get_handle(),
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

                render_tsm_pass_begin_info.framebuffer = fbos.trans_sm[pong_index].fb;
                vkCmdBeginRenderPass(draw_cmd_buffers[i], &render_tsm_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);
                {
                    // Viewport
                    VkViewport viewport = vkb::initializers::viewport((float) width / TSM_UPSAMPLE_RATIO, (float) height / TSM_UPSAMPLE_RATIO, 0.0f, 1.0f);
                    vkCmdSetViewport(draw_cmd_buffers[i], 0, 1, &viewport);

                    // Scissor
                    VkRect2D scissor = vkb::initializers::rect2D(width / TSM_UPSAMPLE_RATIO, height / TSM_UPSAMPLE_RATIO, 0, 0);
                    vkCmdSetScissor(draw_cmd_buffers[i], 0, 1, &scissor);

                    // Pipeline layout
                    vkCmdBindDescriptorSets(draw_cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layouts.trans_sm, 0, 1, &descriptor_sets.trans_sm[pong_index], 0, nullptr);

                    // Draw
                    vkCmdBindPipeline(draw_cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.trans_sm);
                    vkCmdBindVertexBuffers(draw_cmd_buffers[i], 0, 1, model.vertex_buffer->get(), offsets);
                    vkCmdBindIndexBuffer(draw_cmd_buffers[i], model.index_buffer->get_handle(), 0, VK_INDEX_TYPE_UINT32);                
                    vkCmdDrawIndexed(draw_cmd_buffers[i], model.index_count, 1, 0, 0, 0);
                }
                vkCmdEndRenderPass(draw_cmd_buffers[i]);

                vkb::insert_image_memory_barrier(
                    draw_cmd_buffers[i],
                    fbos.trans_sm[ping_index].images[0].get_handle(),
                    VK_ACCESS_SHADER_READ_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

                vkb::insert_image_memory_barrier(
                        draw_cmd_buffers[i],
                        fbos.trans_sm[pong_index].images[0].get_handle(),
                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
            }
            else
            {
                VkClearColorValue clear_color{{0.0f, 0.0f, 0.0f, 1.0f}};
                VkImageSubresourceRange subresource_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

                vkCmdClearColorImage(
                    draw_cmd_buffers[i],
                    fbos.trans_sm[pong_index].images[0].get_handle(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    &clear_color,
                    1,
                    &subresource_range);

                vkb::insert_image_memory_barrier(
                    draw_cmd_buffers[i],
                    fbos.trans_sm[pong_index].images[0].get_handle(),
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
            }

            // Copy image to texture
            {

                vkb::insert_image_memory_barrier(
                    draw_cmd_buffers[i],
                    tsm_texture.image,
                    0,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_HOST_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

                VkImageCopy image_copy                   = {};
                image_copy.extent                        = VkExtent3D{width / TSM_UPSAMPLE_RATIO, height / TSM_UPSAMPLE_RATIO, 1};
                image_copy.srcOffset                     = {0, 0, 0};
                image_copy.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
                image_copy.srcSubresource.mipLevel       = 0;
                image_copy.srcSubresource.baseArrayLayer = 0;
                image_copy.srcSubresource.layerCount     = 1;
                image_copy.dstOffset                     = {0, 0, 0};
                image_copy.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
                image_copy.dstSubresource.mipLevel       = 0;
                image_copy.dstSubresource.baseArrayLayer = 0;
                image_copy.dstSubresource.layerCount     = 1;

                vkCmdCopyImage(
                    draw_cmd_buffers[i],
                    fbos.trans_sm[pong_index].images[0].get_handle(),
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    tsm_texture.image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1,
                    &image_copy);

                vkb::insert_image_memory_barrier(
                    draw_cmd_buffers[i],
                    tsm_texture.image,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
            }

            vkb::insert_image_memory_barrier(
                draw_cmd_buffers[i],
                fbos.trans_sm[pong_index].images[0].get_handle(),
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

            // Begin render pass (deferred shading)
            render_deferred_pass_begin_info.framebuffer = fbos.deferred.fb;
            vkCmdBeginRenderPass(draw_cmd_buffers[i], &render_deferred_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);
            {
                // Viewport
                VkViewport viewport = vkb::initializers::viewport((float) width, (float) height, 0.0f, 1.0f);
                vkCmdSetViewport(draw_cmd_buffers[i], 0, 1, &viewport);

                // Scissor
                VkRect2D scissor = vkb::initializers::rect2D(width, height, 0, 0);
                vkCmdSetScissor(draw_cmd_buffers[i], 0, 1, &scissor);

                // Pipeline layout
                vkCmdBindDescriptorSets(draw_cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layouts.deferred, 0, 1, &descriptor_sets.deferred, 0, nullptr);

                // Background
                vkCmdBindPipeline(draw_cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.background);
                vkCmdBindVertexBuffers(draw_cmd_buffers[i], 0, 1, cube.vertex_buffer->get(), offsets);
                vkCmdBindIndexBuffer(draw_cmd_buffers[i], cube.index_buffer->get_handle(), 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(draw_cmd_buffers[i], cube.index_count, 1, 0, 0, 0);

                // Object
                vkCmdBindPipeline(draw_cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.deferred);
                vkCmdBindVertexBuffers(draw_cmd_buffers[i], 0, 1, model.vertex_buffer->get(), offsets);
                vkCmdBindIndexBuffer(draw_cmd_buffers[i], model.index_buffer->get_handle(), 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(draw_cmd_buffers[i], model.index_count, 1, 0, 0, 0);
            }
            // End render pass (camera pass)
            vkCmdEndRenderPass(draw_cmd_buffers[i]);

            // Change image layout (color : deferred)
            vkb::insert_image_memory_barrier(
                draw_cmd_buffers[i],
                fbos.deferred.images[0].get_handle(),
                0,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

            // Postprocess
            render_postprocess_begin_info.framebuffer = framebuffers[i];
            vkCmdBeginRenderPass(draw_cmd_buffers[i], &render_postprocess_begin_info, VK_SUBPASS_CONTENTS_INLINE);
            {
                // Viewport
                VkViewport viewport = vkb::initializers::viewport((float) width, (float) height, 0.0f, 1.0f);
                vkCmdSetViewport(draw_cmd_buffers[i], 0, 1, &viewport);

                // Scissor
                VkRect2D scissor = vkb::initializers::rect2D(width, height, 0, 0);
                vkCmdSetScissor(draw_cmd_buffers[i], 0, 1, &scissor);

                // Pipeline layout
                vkCmdBindDescriptorSets(draw_cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layouts.postprocess, 0, 1, &descriptor_sets.postprocess, 0, nullptr);

                // Draw
                vkCmdBindPipeline(draw_cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.postprocess);
                vkCmdBindVertexBuffers(draw_cmd_buffers[i], 0, 1, rect.vertex_buffer->get(), offsets);
                vkCmdBindIndexBuffer(draw_cmd_buffers[i], rect.index_buffer->get_handle(), 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(draw_cmd_buffers[i], rect.index_count, 1, 0, 0, 0);

                // UI
                draw_ui(draw_cmd_buffers[i]);
            }
            // End render pass (postprocess)
            vkCmdEndRenderPass(draw_cmd_buffers[i]);
        }
        // END
        VK_CHECK(vkEndCommandBuffer(draw_cmd_buffers[i]));
    }
}

void LinSSScatter::draw()
{
    ApiVulkanSample::prepare_frame();

    // Command buffer to be sumitted to the queue
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers    = &draw_cmd_buffers[current_buffer];

    // Submit to queue
    VK_CHECK(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE));

    ApiVulkanSample::submit_frame();
}

void LinSSScatter::load_model(const std::string &filename)
{
    // Mesh model
    {
        // Load .ply file
        using tinyply::PlyData;
        using tinyply::PlyFile;

        std::unordered_map<LinSSScatterVertexStructure, uint32_t> uniqueVertices;
        std::vector<LinSSScatterVertexStructure>                  vertices;
        std::vector<uint32_t>                                     indices;

        try
        {
            // Open
            std::ifstream reader(filename.c_str(), std::ios::binary);
            if (reader.fail())
            {
                LOGE("Failed to open file: {}", vkb::to_string(filename));
                abort();
            }

            // Read header
            PlyFile file;
            file.parse_header(reader);

            // Request vertex data
            std::shared_ptr<PlyData> vert_data, norm_data, uv_data, face_data;
            try
            {
                vert_data = file.request_properties_from_element("vertex", {"x", "y", "z"});
            }
            catch (const std::invalid_argument &e)
            {
                LOGW("tinyply exception: {}", e.what());
            }

            try
            {
                norm_data = file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
            }
            catch (const std::invalid_argument &e)
            {
                LOGW("tinyply exception: {}", e.what());
            }

            try
            {
                uv_data = file.request_properties_from_element("vertex", {"u", "v"});
            }
            catch (const std::invalid_argument &e)
            {
                LOGW("tinyply exception: {}", e.what());
            }

            try
            {
                face_data = file.request_properties_from_element("face", {"vertex_indices"}, 3);
            }
            catch (const std::invalid_argument &e)
            {
                LOGW("tinyply exception: {}", e.what());
            }

            // Read vertex data
            file.read(reader);

            // Copy vertex data
            const size_t       numVerts = vert_data->count;
            std::vector<float> raw_vertices, raw_normals, raw_uv;
            if (vert_data)
            {
                raw_vertices.resize(numVerts * 3);
                std::memcpy(raw_vertices.data(), vert_data->buffer.get(), sizeof(float) * numVerts * 3);
            }

            if (norm_data)
            {
                raw_normals.resize(numVerts * 3);
                std::memcpy(raw_normals.data(), norm_data->buffer.get(), sizeof(float) * numVerts * 3);
            }

            if (uv_data)
            {
                raw_uv.resize(numVerts * 2);
                std::memcpy(raw_uv.data(), uv_data->buffer.get(), sizeof(float) * numVerts * 2);
            }

            const size_t          numFaces = face_data->count;
            std::vector<uint32_t> raw_indices(numFaces * 3);
            std::memcpy(raw_indices.data(), face_data->buffer.get(), sizeof(uint32_t) * numFaces * 3);

            for (uint32_t i : raw_indices)
            {
                glm::vec3 pos, normal;
                glm::vec2 uv;

                if (vert_data)
                {
                    pos = glm::vec3(raw_vertices[i * 3 + 0], raw_vertices[i * 3 + 1], raw_vertices[i * 3 + 2]);
                    uv  = glm::vec2(pos.x, pos.y) * 0.5f + 0.5f;        // TODO: uv is simply computed from vertex position.
                }

                if (norm_data)
                {
                    normal = glm::vec3(raw_normals[i * 3 + 0], raw_normals[i * 3 + 1], raw_normals[i * 3 + 2]);
                }

                LinSSScatterVertexStructure vtx(pos, uv, normal);
                if (uniqueVertices.count(vtx) == 0)
                {
                    uniqueVertices[vtx] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vtx);
                }
                indices.push_back(uniqueVertices[vtx]);
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
        }

        // Create vertex/index buffers
        model.index_count       = static_cast<uint32_t>(indices.size());
        auto vertex_buffer_size = vkb::to_u32(vertices.size() * sizeof(LinSSScatterVertexStructure));
        auto index_buffer_size  = vkb::to_u32(indices.size() * sizeof(uint32_t));

        // Create buffers
        // For the sake of simplicity we won't stage the vertex data to the gpu memory
        // Vertex buffer
        model.vertex_buffer = std::make_unique<vkb::core::Buffer>(get_device(),
                                                                  vertex_buffer_size,
                                                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                                  VMA_MEMORY_USAGE_CPU_TO_GPU);
        model.vertex_buffer->update(vertices.data(), vertex_buffer_size);

        model.index_buffer = std::make_unique<vkb::core::Buffer>(get_device(),
                                                                 index_buffer_size,
                                                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                                 VMA_MEMORY_USAGE_CPU_TO_GPU);
        model.index_buffer->update(indices.data(), index_buffer_size);
    }
}

void LinSSScatter::prepare_primitive_objects()
{
    // Rect
    {
        static const float rect_vertices[4][3] = {
            {-1.0f, -1.0f, 0.0f},
            {-1.0f, 1.0f, 0.0f},
            {1.0f, -1.0f, 0.0f},
            {1.0f, 1.0f, 0.0f}};

        static const uint32_t rect_indices[2][3] = {
            {0, 3, 1},
            {0, 2, 3}};

        rect.index_count            = 2 * 3;
        uint32_t vertex_buffer_size = sizeof(float) * 4 * 3;
        uint32_t index_buffer_size  = sizeof(uint32_t) * 2 * 3;

        rect.vertex_buffer = std::make_unique<vkb::core::Buffer>(get_device(),
                                                                 vertex_buffer_size,
                                                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                                 VMA_MEMORY_USAGE_CPU_TO_GPU);
        rect.vertex_buffer->update((uint8_t *) &rect_vertices[0][0], vertex_buffer_size);

        rect.index_buffer = std::make_unique<vkb::core::Buffer>(get_device(),
                                                                index_buffer_size,
                                                                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                                VMA_MEMORY_USAGE_CPU_TO_GPU);
        rect.index_buffer->update((uint8_t *) &rect_indices[0][0], index_buffer_size);
    }

    // Cube
    {
        static const float cube_vertices[8][3] = {
            {-1.0f, -1.0f, -1.0f},
            {-1.0f, -1.0f, 1.0f},
            {-1.0f, 1.0f, -1.0f},
            {1.0f, -1.0f, -1.0f},
            {-1.0f, 1.0f, 1.0f},
            {1.0f, -1.0f, 1.0f},
            {1.0f, 1.0f, -1.0f},
            {1.0f, 1.0f, 1.0f}};

        static const uint32_t cube_indices[12][3] = {
            {0, 6, 2},
            {0, 3, 6},
            {1, 7, 5},
            {1, 4, 7},
            {0, 4, 1},
            {0, 2, 4},
            {3, 7, 6},
            {3, 5, 7},
            {0, 5, 3},
            {0, 1, 5},
            {2, 7, 4},
            {2, 6, 7}};

        cube.index_count            = 12 * 3;
        uint32_t vertex_buffer_size = sizeof(float) * 8 * 3;
        uint32_t index_buffer_size  = sizeof(uint32_t) * 12 * 3;

        cube.vertex_buffer = std::make_unique<vkb::core::Buffer>(get_device(),
                                                                 vertex_buffer_size,
                                                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                                 VMA_MEMORY_USAGE_CPU_TO_GPU);
        cube.vertex_buffer->update((uint8_t *) &cube_vertices[0][0], vertex_buffer_size);

        cube.index_buffer = std::make_unique<vkb::core::Buffer>(get_device(),
                                                                index_buffer_size,
                                                                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                                VMA_MEMORY_USAGE_CPU_TO_GPU);
        cube.index_buffer->update((uint8_t *) &cube_indices[0][0], index_buffer_size);
    }
}

void LinSSScatter::prepare_bssrdf(const std::string &filename)
{
    // Load BSSRDF
    std::ifstream reader(filename.c_str(), std::ios::in | std::ios::binary);
    if (reader.fail())
    {
        LOGE("Failed to open file: {}", filename);
    }

    uint32_t area_width, area_height, n_gauss, ksize;
    reader.read((char *) &area_width, sizeof(uint32_t));
    reader.read((char *) &area_height, sizeof(uint32_t));
    reader.read((char *) &n_gauss, sizeof(uint32_t));
    reader.read((char *) &ksize, sizeof(uint32_t));

    // Load weights
    auto buffer = std::make_unique<double[]>(n_gauss * 3);
    auto data_W = std::make_unique<float[]>(area_width * area_height * n_gauss * 4);
    for (int y = 0; y < area_height; y++)
    {
        for (int x = 0; x < area_width; x++)
        {
            reader.read((char *) buffer.get(), sizeof(double) * n_gauss * 3);
            for (int h = 0; h < n_gauss; h++)
            {
                const int idx       = (h * area_height + (area_height - y - 1)) * area_width + x;
                data_W[idx * 4 + 0] = std::max(0.0f, static_cast<float>(buffer[h * 3 + 0]));
                data_W[idx * 4 + 1] = std::max(0.0f, static_cast<float>(buffer[h * 3 + 1]));
                data_W[idx * 4 + 2] = std::max(0.0f, static_cast<float>(buffer[h * 3 + 2]));
                data_W[idx * 4 + 3] = 1.0f;
            }
        }
    }

    // Load beta (inverse of sigma)
    bssrdf.sigmas.resize(n_gauss);
    reader.read((char *) buffer.get(), sizeof(double) * n_gauss * 3);
    for (int i = 0; i < n_gauss; i++)
    {
        const float r       = static_cast<float>(buffer[i * 3 + 0]);
        const float g       = static_cast<float>(buffer[i * 3 + 1]);
        const float b       = static_cast<float>(buffer[i * 3 + 2]);
        const float sigma_r = std::sqrt(1.0f / std::max(1.0e-4f, r));
        const float sigma_g = std::sqrt(1.0f / std::max(1.0e-4f, g));
        const float sigma_b = std::sqrt(1.0f / std::max(1.0e-4f, b));
        bssrdf.sigmas[i]    = glm::vec4(sigma_r, sigma_g, sigma_b, 1.0f);
    }
    bssrdf.width   = area_width;
    bssrdf.height  = area_height;
    bssrdf.n_gauss = n_gauss;
    bssrdf.ksize   = ksize;

    reader.close();

    // Copy weights
    auto data_G_ast_W = std::make_unique<float[]>(area_width * area_height * n_gauss * 4);
    std::memcpy(data_G_ast_W.get(), data_W.get(), sizeof(float) * area_width * area_height * n_gauss * 4);

    // Apply Guassian filter to weight maps
    for (int i = 0; i < n_gauss; i++)
    {
        gaussBlur(&data_G_ast_W[i * area_width * area_height * 4], bssrdf.sigmas[i], area_width, area_height, 4);
    }

    // 3D texture support in Vulkan is mandatory (in contrast to OpenGL) so no need to check if it's supported
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(get_device().get_gpu().get_handle(), VK_FORMAT_R32G32B32A32_SFLOAT, &formatProperties);
    // Check if format supports transfer
    if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_DST_BIT))
    {
        LOGE("Error: Device does not support flag TRANSFER_DST for selected texture format!");
    }

    // Check if GPU supports requested 3D texture dimensions
    uint32_t maxImageDimension3D(get_device().get_gpu().get_properties().limits.maxImageDimension3D);
    if (area_width > maxImageDimension3D || area_height > maxImageDimension3D || n_gauss > maxImageDimension3D)
    {
        LOGE("Error: Requested texture dimensions is greater than supported 3D texture dimension!");
    }

    // Prepare staging buffer
    VkMemoryAllocateInfo memory_allocate_info = vkb::initializers::memory_allocate_info();
    VkMemoryRequirements memory_requirements  = {};

    VkBuffer       staging_buffer;
    VkDeviceMemory staging_memory;

    VkBufferCreateInfo buffer_create_info = vkb::initializers::buffer_create_info();
    buffer_create_info.size               = area_width * area_height * n_gauss * sizeof(float) * 4;
    buffer_create_info.usage              = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buffer_create_info.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(get_device().get_handle(), &buffer_create_info, nullptr, &staging_buffer));

    vkGetBufferMemoryRequirements(get_device().get_handle(), staging_buffer, &memory_requirements);
    memory_allocate_info.allocationSize  = memory_requirements.size;
    memory_allocate_info.memoryTypeIndex = get_device().get_memory_type(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(get_device().get_handle(), &memory_allocate_info, nullptr, &staging_memory));
    VK_CHECK(vkBindBufferMemory(get_device().get_handle(), staging_buffer, staging_memory, 0));

    // 3D texture for blurred weights
    {
        // Copy CPU data to GPU
        uint8_t *data;
        VK_CHECK(vkMapMemory(get_device().get_handle(), staging_memory, 0, memory_requirements.size, 0, (void **) &data));

        uint32_t texture_size = area_width * area_height * n_gauss * sizeof(float) * 4;
        std::memcpy(data, data_W.get(), texture_size);
        vkUnmapMemory(get_device().get_handle(), staging_memory);

        // Setup buffer copy region
        VkBufferImageCopy buffer_copy_region               = {};
        buffer_copy_region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        buffer_copy_region.imageSubresource.mipLevel       = 0;
        buffer_copy_region.imageSubresource.baseArrayLayer = 0;
        buffer_copy_region.imageSubresource.layerCount     = 1;
        buffer_copy_region.imageExtent.width               = area_width;
        buffer_copy_region.imageExtent.height              = area_height;
        buffer_copy_region.imageExtent.depth               = n_gauss;
        buffer_copy_region.bufferOffset                    = 0;

        VkImageCreateInfo image_create_info = vkb::initializers::image_create_info();
        image_create_info.imageType         = VK_IMAGE_TYPE_3D;
        image_create_info.format            = VK_FORMAT_R32G32B32A32_SFLOAT;
        image_create_info.mipLevels         = 1;
        image_create_info.arrayLayers       = 1;
        image_create_info.samples           = VK_SAMPLE_COUNT_1_BIT;
        image_create_info.tiling            = VK_IMAGE_TILING_OPTIMAL;
        image_create_info.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
        image_create_info.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        image_create_info.extent            = {area_width, area_height, n_gauss};
        image_create_info.usage             = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        VK_CHECK(vkCreateImage(get_device().get_handle(), &image_create_info, nullptr, &bssrdf.image_W));

        vkGetImageMemoryRequirements(get_device().get_handle(), bssrdf.image_W, &memory_requirements);
        memory_allocate_info.allocationSize  = memory_requirements.size;
        memory_allocate_info.memoryTypeIndex = get_device().get_memory_type(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(get_device().get_handle(), &memory_allocate_info, nullptr, &bssrdf.device_memory_W));
        VK_CHECK(vkBindImageMemory(get_device().get_handle(), bssrdf.image_W, bssrdf.device_memory_W, 0));

        VkCommandBuffer copy_command = device->create_command_buffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

        vkb::insert_image_memory_barrier(
            copy_command,
            bssrdf.image_W,
            0,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

        vkCmdCopyBufferToImage(
            copy_command,
            staging_buffer,
            bssrdf.image_W,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &buffer_copy_region);

        vkb::insert_image_memory_barrier(
            copy_command,
            bssrdf.image_W,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

        device->flush_command_buffer(copy_command, queue, true);

        // Image view
        VkImageViewCreateInfo view_create_info           = vkb::initializers::image_view_create_info();
        view_create_info.image                           = bssrdf.image_W;
        view_create_info.viewType                        = VK_IMAGE_VIEW_TYPE_3D;
        view_create_info.format                          = VK_FORMAT_R32G32B32A32_SFLOAT;
        view_create_info.components                      = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
        view_create_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        view_create_info.subresourceRange.baseMipLevel   = 0;
        view_create_info.subresourceRange.levelCount     = 1;
        view_create_info.subresourceRange.baseArrayLayer = 0;
        view_create_info.subresourceRange.layerCount     = 1;
        VK_CHECK(vkCreateImageView(get_device().get_handle(), &view_create_info, nullptr, &bssrdf.view_W));
    }

    // 3D texture for blurred weights
    {
        // Copy CPU data to GPU
        uint8_t *data;
        VK_CHECK(vkMapMemory(get_device().get_handle(), staging_memory, 0, memory_requirements.size, 0, (void **) &data));

        uint32_t texture_size = area_width * area_height * n_gauss * sizeof(float) * 4;
        std::memcpy(data, data_G_ast_W.get(), texture_size);
        vkUnmapMemory(get_device().get_handle(), staging_memory);

        // Setup buffer copy region
        VkBufferImageCopy buffer_copy_region               = {};
        buffer_copy_region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        buffer_copy_region.imageSubresource.mipLevel       = 0;
        buffer_copy_region.imageSubresource.baseArrayLayer = 0;
        buffer_copy_region.imageSubresource.layerCount     = 1;
        buffer_copy_region.imageExtent.width               = area_width;
        buffer_copy_region.imageExtent.height              = area_height;
        buffer_copy_region.imageExtent.depth               = n_gauss;
        buffer_copy_region.bufferOffset                    = 0;

        VkImageCreateInfo image_create_info = vkb::initializers::image_create_info();
        image_create_info.imageType         = VK_IMAGE_TYPE_3D;
        image_create_info.format            = VK_FORMAT_R32G32B32A32_SFLOAT;
        image_create_info.mipLevels         = 1;
        image_create_info.arrayLayers       = 1;
        image_create_info.samples           = VK_SAMPLE_COUNT_1_BIT;
        image_create_info.tiling            = VK_IMAGE_TILING_OPTIMAL;
        image_create_info.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
        image_create_info.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        image_create_info.extent            = {area_width, area_height, n_gauss};
        image_create_info.usage             = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        VK_CHECK(vkCreateImage(get_device().get_handle(), &image_create_info, nullptr, &bssrdf.image_G_ast_W));

        vkGetImageMemoryRequirements(get_device().get_handle(), bssrdf.image_G_ast_W, &memory_requirements);
        memory_allocate_info.allocationSize  = memory_requirements.size;
        memory_allocate_info.memoryTypeIndex = get_device().get_memory_type(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(get_device().get_handle(), &memory_allocate_info, nullptr, &bssrdf.device_memory_G_ast_W));
        VK_CHECK(vkBindImageMemory(get_device().get_handle(), bssrdf.image_G_ast_W, bssrdf.device_memory_G_ast_W, 0));

        VkCommandBuffer copy_command = device->create_command_buffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

        vkb::insert_image_memory_barrier(
            copy_command,
            bssrdf.image_G_ast_W,
            0,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

        vkCmdCopyBufferToImage(
            copy_command,
            staging_buffer,
            bssrdf.image_G_ast_W,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &buffer_copy_region);

        vkb::insert_image_memory_barrier(
            copy_command,
            bssrdf.image_G_ast_W,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

        device->flush_command_buffer(copy_command, queue, true);

        // Image view
        VkImageViewCreateInfo view_create_info           = vkb::initializers::image_view_create_info();
        view_create_info.image                           = bssrdf.image_G_ast_W;
        view_create_info.viewType                        = VK_IMAGE_VIEW_TYPE_3D;
        view_create_info.format                          = VK_FORMAT_R32G32B32A32_SFLOAT;
        view_create_info.components                      = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
        view_create_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        view_create_info.subresourceRange.baseMipLevel   = 0;
        view_create_info.subresourceRange.levelCount     = 1;
        view_create_info.subresourceRange.baseArrayLayer = 0;
        view_create_info.subresourceRange.layerCount     = 1;
        VK_CHECK(vkCreateImageView(get_device().get_handle(), &view_create_info, nullptr, &bssrdf.view_G_ast_W));
    }

    // Clean up staging resources
    vkFreeMemory(get_device().get_handle(), staging_memory, nullptr);
    vkDestroyBuffer(get_device().get_handle(), staging_buffer, nullptr);

    // Sampler
    VkSamplerCreateInfo sampler_create_info = vkb::initializers::sampler_create_info();
    sampler_create_info.magFilter           = VK_FILTER_LINEAR;
    sampler_create_info.minFilter           = VK_FILTER_LINEAR;
    sampler_create_info.mipmapMode          = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_create_info.addressModeU        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_create_info.addressModeV        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_create_info.addressModeW        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_create_info.mipLodBias          = 0.0f;
    sampler_create_info.compareOp           = VK_COMPARE_OP_NEVER;
    sampler_create_info.minLod              = 0.0f;
    sampler_create_info.maxLod              = 1.0f;
    sampler_create_info.borderColor         = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    if (get_device().get_gpu().get_features().samplerAnisotropy)
    {
        // Use max. level of anisotropy for this example
        sampler_create_info.maxAnisotropy    = get_device().get_gpu().get_properties().limits.maxSamplerAnisotropy;
        sampler_create_info.anisotropyEnable = VK_TRUE;
    }
    else
    {
        // The device does not support anisotropic filtering
        sampler_create_info.maxAnisotropy    = 1.0;
        sampler_create_info.anisotropyEnable = VK_FALSE;
    }
    VK_CHECK(vkCreateSampler(device->get_handle(), &sampler_create_info, nullptr, &bssrdf.sampler));

    // Print information
    for (int i = 0; i < bssrdf.n_gauss; i++)
    {
        LOGI("BSSRDF sigma[{}]: {}", vkb::to_string(i), glm::to_string(bssrdf.sigmas[i]));
    }
}

void LinSSScatter::setup_descriptor_set_layout()
{
    // Light pass
    {
        std::vector<VkDescriptorSetLayoutBinding> set_layout_bindings =
            {
                // Binding 0 : Vertex shader uniform buffer
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    0)};

        VkDescriptorSetLayoutCreateInfo descriptor_layout_create_info =
            vkb::initializers::descriptor_set_layout_create_info(
                set_layout_bindings.data(),
                static_cast<uint32_t>(set_layout_bindings.size()));

        VK_CHECK(vkCreateDescriptorSetLayout(get_device().get_handle(), &descriptor_layout_create_info, nullptr, &descriptor_set_layouts.light_pass));

        VkPipelineLayoutCreateInfo pipeline_layout_create_info =
            vkb::initializers::pipeline_layout_create_info(
                &descriptor_set_layouts.light_pass,
                1);

        VK_CHECK(vkCreatePipelineLayout(get_device().get_handle(), &pipeline_layout_create_info, nullptr, &pipeline_layouts.light_pass));
    }

    // Direct pass
    {
        std::vector<VkDescriptorSetLayoutBinding> set_layout_bindings =
            {
                // Binding 0 : Vertex shader uniform buffer
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    0),
                // Binding 1 : Fragment shader uniform buffer
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    1),
                // Binding 2 : Fragment shader image sampler (Ks)
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    2),
                // Binding 3 : Fragment shader image sampler (envmap)
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    3),
                // Binding 4 : Fragment shader depth buffer
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    4)};

        VkDescriptorSetLayoutCreateInfo descriptor_layout_create_info =
            vkb::initializers::descriptor_set_layout_create_info(
                set_layout_bindings.data(),
                static_cast<uint32_t>(set_layout_bindings.size()));

        VK_CHECK(vkCreateDescriptorSetLayout(get_device().get_handle(), &descriptor_layout_create_info, nullptr, &descriptor_set_layouts.direct_pass));

        VkPipelineLayoutCreateInfo pipeline_layout_create_info =
            vkb::initializers::pipeline_layout_create_info(
                &descriptor_set_layouts.direct_pass,
                1);

        VK_CHECK(vkCreatePipelineLayout(get_device().get_handle(), &pipeline_layout_create_info, nullptr, &pipeline_layouts.direct_pass));
    }

    // Gaussian filter
    {
        std::vector<VkDescriptorSetLayoutBinding> set_layout_bindings =
            {
                // Binding 0 : input image storage
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    0),
                // Binding 0 : output image storage
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    1),
                // Binding 1 : buffer image
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    2),
                // Binding 2 : uniform buffer object
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    3),
                // Binding 3-5 : image samplers (position, normal, depth)
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    4),
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    5),
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    6)};

        VkDescriptorSetLayoutCreateInfo descriptor_layout_create_info =
            vkb::initializers::descriptor_set_layout_create_info(
                set_layout_bindings.data(),
                static_cast<uint32_t>(set_layout_bindings.size()));

        VK_CHECK(vkCreateDescriptorSetLayout(get_device().get_handle(), &descriptor_layout_create_info, nullptr, &descriptor_set_layouts.gauss_filter));

        VkPipelineLayoutCreateInfo pipeline_layout_create_info =
            vkb::initializers::pipeline_layout_create_info(
                &descriptor_set_layouts.gauss_filter,
                1);

        VK_CHECK(vkCreatePipelineLayout(get_device().get_handle(), &pipeline_layout_create_info, nullptr, &pipeline_layouts.gauss_filter));
    }

    // LinSSS accumulation
    {
        std::vector<VkDescriptorSetLayoutBinding> set_layout_bindings =
            {
                // Binding 0 : destination image storage
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    0),
                // Binding 1 : uniform buffer object
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    1),
                // Binding 2 : 3D image sampler for W
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    2),
                // Binding 3 : 3D image sampler for (G * W)
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    3),
                // Binding 4 : 2D image sampler for (G * Phi)
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    4),
                // Binding 5, 6, 7 : 2D image sampler for position, normal, and depth
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    5),
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    6),
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    7)};

        VkDescriptorSetLayoutCreateInfo descriptor_layout_create_info =
            vkb::initializers::descriptor_set_layout_create_info(
                set_layout_bindings.data(),
                static_cast<uint32_t>(set_layout_bindings.size()));

        VK_CHECK(vkCreateDescriptorSetLayout(get_device().get_handle(), &descriptor_layout_create_info, nullptr, &descriptor_set_layouts.linsss));

        VkPipelineLayoutCreateInfo pipeline_layout_create_info =
            vkb::initializers::pipeline_layout_create_info(
                &descriptor_set_layouts.linsss,
                1);

        VK_CHECK(vkCreatePipelineLayout(get_device().get_handle(), &pipeline_layout_create_info, nullptr, &pipeline_layouts.linsss));
    }

    // Translucent shadow maps
    {
        std::vector<VkDescriptorSetLayoutBinding> set_layout_bindings =
            {
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    0),
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    1),
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    2),
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    3),
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    4),
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    5),
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    6),
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    7)};

        VkDescriptorSetLayoutCreateInfo descriptor_layout_create_info =
            vkb::initializers::descriptor_set_layout_create_info(
                set_layout_bindings.data(),
                static_cast<uint32_t>(set_layout_bindings.size()));

        VK_CHECK(vkCreateDescriptorSetLayout(get_device().get_handle(), &descriptor_layout_create_info, nullptr, &descriptor_set_layouts.trans_sm));

        VkPipelineLayoutCreateInfo pipeline_layout_create_info =
            vkb::initializers::pipeline_layout_create_info(
                &descriptor_set_layouts.trans_sm,
                1);

        VK_CHECK(vkCreatePipelineLayout(get_device().get_handle(), &pipeline_layout_create_info, nullptr, &pipeline_layouts.trans_sm));
    }

    // Deferred shading
    {
        std::vector<VkDescriptorSetLayoutBinding> set_layout_bindings =
            {
                // Binding 0 : Vertex shader uniform buffer
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    0),
                // Binding 1 : Fragment shader uniform buffer
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    1),
                // Binding 2 : Fragment shader image sampler (envmap)
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    2),
                // Binding 3 : Fragment shader image sampler (sss)
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    3),
                // Binding 4 : Fragment shader image sampler (tsm)
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    4),
                // Binding 5 : Fragment shader specular texture
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    5),
                // Binding 6 : Fragment shader depth texture
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    6)};

        VkDescriptorSetLayoutCreateInfo descriptor_layout_create_info =
            vkb::initializers::descriptor_set_layout_create_info(
                set_layout_bindings.data(),
                static_cast<uint32_t>(set_layout_bindings.size()));

        VK_CHECK(vkCreateDescriptorSetLayout(get_device().get_handle(), &descriptor_layout_create_info, nullptr, &descriptor_set_layouts.deferred));

        VkPipelineLayoutCreateInfo pipeline_layout_create_info =
            vkb::initializers::pipeline_layout_create_info(
                &descriptor_set_layouts.deferred,
                1);

        VK_CHECK(vkCreatePipelineLayout(get_device().get_handle(), &pipeline_layout_create_info, nullptr, &pipeline_layouts.deferred));
    }

    // Postprocess
    {
        std::vector<VkDescriptorSetLayoutBinding> set_layout_bindings =
            {
                // Binding 0 : Vertex shader uniform buffer
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    0),
                // Binding 1 : Fragment shader image sampler
                vkb::initializers::descriptor_set_layout_binding(
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    1)};

        VkDescriptorSetLayoutCreateInfo descriptor_layout_create_info =
            vkb::initializers::descriptor_set_layout_create_info(
                set_layout_bindings.data(),
                static_cast<uint32_t>(set_layout_bindings.size()));

        VK_CHECK(vkCreateDescriptorSetLayout(get_device().get_handle(), &descriptor_layout_create_info, nullptr, &descriptor_set_layouts.postprocess));

        VkPipelineLayoutCreateInfo pipeline_layout_create_info =
            vkb::initializers::pipeline_layout_create_info(
                &descriptor_set_layouts.postprocess,
                1);

        VK_CHECK(vkCreatePipelineLayout(get_device().get_handle(), &pipeline_layout_create_info, nullptr, &pipeline_layouts.postprocess));
    }
}

void LinSSScatter::setup_descriptor_set()
{
    // Light pass
    {
        // Descriptor pool
        std::vector<VkDescriptorPoolSize> pool_sizes =
            {
                vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1)};

        VkDescriptorPoolCreateInfo descriptor_pool_create_info =
            vkb::initializers::descriptor_pool_create_info(
                static_cast<uint32_t>(pool_sizes.size()),
                pool_sizes.data(),
                1);

        VK_CHECK(vkCreateDescriptorPool(get_device().get_handle(), &descriptor_pool_create_info, nullptr, &descriptor_pools.light_pass));

        // Memory allocation for descriptor set
        VkDescriptorSetAllocateInfo alloc_info =
            vkb::initializers::descriptor_set_allocate_info(
                descriptor_pools.light_pass,
                &descriptor_set_layouts.light_pass,
                1);

        VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &alloc_info, &descriptor_sets.light_pass));
    }

    // Direct pass
    {
        // Descriptor pool
        std::vector<VkDescriptorPoolSize> pool_sizes =
            {
                vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
                vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3)};

        VkDescriptorPoolCreateInfo descriptor_pool_create_info =
            vkb::initializers::descriptor_pool_create_info(
                static_cast<uint32_t>(pool_sizes.size()),
                pool_sizes.data(),
                1);

        VK_CHECK(vkCreateDescriptorPool(get_device().get_handle(), &descriptor_pool_create_info, nullptr, &descriptor_pools.direct_pass));

        // Memory allocation for descriptor set
        VkDescriptorSetAllocateInfo alloc_info =
            vkb::initializers::descriptor_set_allocate_info(
                descriptor_pools.direct_pass,
                &descriptor_set_layouts.direct_pass,
                1);

        VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &alloc_info, &descriptor_sets.direct_pass));
    }

    // Gaussian filter
    {
        // Descriptor pool
        std::vector<VkDescriptorPoolSize> pool_sizes =
            {
                vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 * MAX_MIP_LEVELS * 2),
                vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * MAX_MIP_LEVELS * 2),
                vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * MAX_MIP_LEVELS * 2)};

        VkDescriptorPoolCreateInfo descriptor_pool_create_info =
            vkb::initializers::descriptor_pool_create_info(
                static_cast<uint32_t>(pool_sizes.size()),
                pool_sizes.data(),
                MAX_MIP_LEVELS * 2);

        VK_CHECK(vkCreateDescriptorPool(get_device().get_handle(), &descriptor_pool_create_info, nullptr, &descriptor_pools.gauss_filter));

        // Memory allocation for descriptor set
        VkDescriptorSetAllocateInfo alloc_info =
            vkb::initializers::descriptor_set_allocate_info(
                descriptor_pools.gauss_filter,
                &descriptor_set_layouts.gauss_filter,
                1);

        descriptor_sets.gauss_horz_filter.resize(MAX_MIP_LEVELS);
        descriptor_sets.gauss_vert_filter.resize(MAX_MIP_LEVELS);
        for (uint32_t i = 0; i < MAX_MIP_LEVELS; i++)
        {
            VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &alloc_info, &descriptor_sets.gauss_horz_filter[i]));
            VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &alloc_info, &descriptor_sets.gauss_vert_filter[i]));
        }
    }

    // LinSSS accumulation
    {
        // Descriptor pool
        std::vector<VkDescriptorPoolSize> pool_sizes = {
            vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
            vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),
            vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6)};

        VkDescriptorPoolCreateInfo descriptor_pool_create_info =
            vkb::initializers::descriptor_pool_create_info(
                static_cast<uint32_t>(pool_sizes.size()),
                pool_sizes.data(),
                1);

        VK_CHECK(vkCreateDescriptorPool(get_device().get_handle(), &descriptor_pool_create_info, nullptr, &descriptor_pools.linsss));

        // Memory allocation for descriptor set
        VkDescriptorSetAllocateInfo alloc_info =
            vkb::initializers::descriptor_set_allocate_info(
                descriptor_pools.linsss,
                &descriptor_set_layouts.linsss,
                1);

        VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &alloc_info, &descriptor_sets.linsss));
    }

    // Translucent shadow maps
    {
        // Descriptor pool
        std::vector<VkDescriptorPoolSize> pool_sizes =
            {
                vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 * 2),
                vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5 * 2)};

        VkDescriptorPoolCreateInfo descriptor_pool_create_info =
            vkb::initializers::descriptor_pool_create_info(
                static_cast<uint32_t>(pool_sizes.size()),
                pool_sizes.data(),
                2);

        VK_CHECK(vkCreateDescriptorPool(get_device().get_handle(), &descriptor_pool_create_info, nullptr, &descriptor_pools.trans_sm));

        // Memory allocation for descriptor set
        VkDescriptorSetAllocateInfo alloc_info =
            vkb::initializers::descriptor_set_allocate_info(
                descriptor_pools.trans_sm,
                &descriptor_set_layouts.trans_sm,
                1);

        VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &alloc_info, &descriptor_sets.trans_sm[0]));
        VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &alloc_info, &descriptor_sets.trans_sm[1]));
    }

    // Deferred shading
    {
        // Descriptor pool
        std::vector<VkDescriptorPoolSize> pool_sizes =
            {
                vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
                vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5)};

        VkDescriptorPoolCreateInfo descriptor_pool_create_info =
            vkb::initializers::descriptor_pool_create_info(
                static_cast<uint32_t>(pool_sizes.size()),
                pool_sizes.data(),
                1);

        VK_CHECK(vkCreateDescriptorPool(get_device().get_handle(), &descriptor_pool_create_info, nullptr, &descriptor_pools.deferred));

        // Memory allocation for descriptor set
        VkDescriptorSetAllocateInfo alloc_info =
            vkb::initializers::descriptor_set_allocate_info(
                descriptor_pools.deferred,
                &descriptor_set_layouts.deferred,
                1);

        VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &alloc_info, &descriptor_sets.deferred));
    }

    // Postprocess
    {
        // Descriptor pool
        std::vector<VkDescriptorPoolSize> pool_sizes =
            {
                vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),
                vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1)};

        VkDescriptorPoolCreateInfo descriptor_pool_create_info =
            vkb::initializers::descriptor_pool_create_info(
                static_cast<uint32_t>(pool_sizes.size()),
                pool_sizes.data(),
                1);

        VK_CHECK(vkCreateDescriptorPool(get_device().get_handle(), &descriptor_pool_create_info, nullptr, &descriptor_pools.postprocess));

        // Memory allocation for descriptor set
        VkDescriptorSetAllocateInfo alloc_info =
            vkb::initializers::descriptor_set_allocate_info(
                descriptor_pools.postprocess,
                &descriptor_set_layouts.postprocess,
                1);

        VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &alloc_info, &descriptor_sets.postprocess));
    }
}

void LinSSScatter::update_descriptor_set()
{
    // Light pass
    {
        VkDescriptorBufferInfo desc_ubo_sm_vs = create_descriptor(*uniform_buffer_sm_vs);

        std::vector<VkWriteDescriptorSet> write_descriptor_sets =
            {
                // Binding 0 : Vertex shader uniform buffer
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.light_pass,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    0,
                    &desc_ubo_sm_vs)};

        vkUpdateDescriptorSets(get_device().get_handle(), static_cast<uint32_t>(write_descriptor_sets.size()), write_descriptor_sets.data(), 0, nullptr);
    }

    // Direct pass
    {
        VkDescriptorBufferInfo desc_ubo_vs = create_descriptor(*uniform_buffer_vs);
        VkDescriptorBufferInfo desc_ubo_fs = create_descriptor(*uniform_buffer_fs);

        VkDescriptorImageInfo desc_envmap_texture;
        desc_envmap_texture.imageView   = envmap_texture.view;
        desc_envmap_texture.sampler     = envmap_texture.sampler;
        desc_envmap_texture.imageLayout = envmap_texture.image_layout;

        VkDescriptorImageInfo desc_Ks_texture;
        desc_Ks_texture.imageView   = Ks_texture.view;
        desc_Ks_texture.sampler     = Ks_texture.sampler;
        desc_Ks_texture.imageLayout = Ks_texture.image_layout;

        VkDescriptorImageInfo desc_depth_buffer;
        desc_depth_buffer.imageView   = fbos.shadow_map.views[3].get_handle();
        desc_depth_buffer.sampler     = fbos.shadow_map.sampler;
        desc_depth_buffer.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        // Descriptor set write information
        std::vector<VkWriteDescriptorSet> write_descriptor_sets =
            {
                // Binding 0 : Vertex shader uniform buffer
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.direct_pass,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    0,
                    &desc_ubo_vs),
                // Binding 1 : Fragment shader uniform buffer
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.direct_pass,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    1,
                    &desc_ubo_fs),
                // Binding 2 : Fragment shader, envmap texture sampler
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.direct_pass,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    2,
                    &desc_Ks_texture),
                // Binding 3 : Fragment shader, envmap texture sampler
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.direct_pass,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    3,
                    &desc_envmap_texture),
                // Biding 4 : Fragment shader, depth buffer sampler
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.direct_pass,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    4,
                    &desc_depth_buffer),
            };

        vkUpdateDescriptorSets(get_device().get_handle(), static_cast<uint32_t>(write_descriptor_sets.size()), write_descriptor_sets.data(), 0, nullptr);
    }

    // Gaussian filter
    {
        const uint32_t mip_levels = max_mip_levels_surface();
        for (uint32_t i = 0; i < mip_levels; i++)
        {
            // Update descriptor set
            VkDescriptorImageInfo desc_in_image;
            desc_in_image.imageView   = in_image_mip_level_views[i];
            desc_in_image.sampler     = VkSampler{};
            desc_in_image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorImageInfo desc_out_image;
            desc_out_image.imageView   = out_image_mip_level_views[i];
            desc_out_image.sampler     = VkSampler{};
            desc_out_image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorImageInfo desc_buf_image;
            desc_buf_image.imageView   = buf_image_mip_level_views[i];
            desc_buf_image.sampler     = VkSampler{};
            desc_buf_image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorImageInfo desc_position_texture;
            desc_position_texture.imageView   = fbos.direct_pass.views[2].get_handle();
            desc_position_texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            desc_position_texture.sampler     = fbos.direct_pass.sampler;

            VkDescriptorImageInfo desc_normal_texture;
            desc_normal_texture.imageView   = fbos.direct_pass.views[3].get_handle();
            desc_normal_texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            desc_normal_texture.sampler     = fbos.direct_pass.sampler;

            VkDescriptorImageInfo desc_depth_texture;
            desc_depth_texture.imageView   = fbos.direct_pass.views[4].get_handle();
            desc_depth_texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            desc_depth_texture.sampler     = fbos.direct_pass.sampler;

            // Update descriptor set
            {
                ubo_gauss_cs.direction = 0;
                uniform_buffer_gauss_horz_cs->convert_and_update(ubo_gauss_cs);
                VkDescriptorBufferInfo desc_ubo_gauss_horz = create_descriptor(*uniform_buffer_gauss_horz_cs);

                std::vector<VkWriteDescriptorSet> write_descriptor_sets =
                    {
                        // Binding 0 : input image
                        vkb::initializers::write_descriptor_set(
                            descriptor_sets.gauss_horz_filter[i],
                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                            0,
                            &desc_in_image),
                        // Binding 0 : input image
                        vkb::initializers::write_descriptor_set(
                            descriptor_sets.gauss_horz_filter[i],
                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                            1,
                            &desc_out_image),
                        // Binding 2 : buffer image
                        vkb::initializers::write_descriptor_set(
                            descriptor_sets.gauss_horz_filter[i],
                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                            2,
                            &desc_buf_image),
                        // Binding 3 : uniform buffer object
                        vkb::initializers::write_descriptor_set(
                            descriptor_sets.gauss_horz_filter[i],
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            3,
                            &desc_ubo_gauss_horz),
                        // Binding 4 : position texture
                        vkb::initializers::write_descriptor_set(
                            descriptor_sets.gauss_horz_filter[i],
                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            4,
                            &desc_position_texture),
                        // Binding 5 : normal texture
                        vkb::initializers::write_descriptor_set(
                            descriptor_sets.gauss_horz_filter[i],
                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            5,
                            &desc_normal_texture),
                        // Binding 6 : depth texture
                        vkb::initializers::write_descriptor_set(
                            descriptor_sets.gauss_horz_filter[i],
                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            6,
                            &desc_depth_texture)};

                vkUpdateDescriptorSets(get_device().get_handle(), static_cast<uint32_t>(write_descriptor_sets.size()), write_descriptor_sets.data(), 0, nullptr);
            }

            {
                ubo_gauss_cs.direction = 1;
                uniform_buffer_gauss_vert_cs->convert_and_update(ubo_gauss_cs);
                VkDescriptorBufferInfo desc_ubo_gauss_vert = create_descriptor(*uniform_buffer_gauss_vert_cs);

                std::vector<VkWriteDescriptorSet> write_descriptor_sets =
                    {
                        // Binding 0 : input image
                        vkb::initializers::write_descriptor_set(
                            descriptor_sets.gauss_vert_filter[i],
                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                            0,
                            &desc_in_image),
                        // Binding 0 : input image
                        vkb::initializers::write_descriptor_set(
                            descriptor_sets.gauss_vert_filter[i],
                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                            1,
                            &desc_out_image),
                        // Binding 2 : buffer image
                        vkb::initializers::write_descriptor_set(
                            descriptor_sets.gauss_vert_filter[i],
                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                            2,
                            &desc_buf_image),
                        // Binding 3 : uniform buffer object
                        vkb::initializers::write_descriptor_set(
                            descriptor_sets.gauss_vert_filter[i],
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            3,
                            &desc_ubo_gauss_vert),
                        // Binding 4 : position texture
                        vkb::initializers::write_descriptor_set(
                            descriptor_sets.gauss_vert_filter[i],
                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            4,
                            &desc_position_texture),
                        // Binding 5 : normal texture
                        vkb::initializers::write_descriptor_set(
                            descriptor_sets.gauss_vert_filter[i],
                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            5,
                            &desc_normal_texture),
                        // Binding 6 : depth texture
                        vkb::initializers::write_descriptor_set(
                            descriptor_sets.gauss_vert_filter[i],
                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            6,
                            &desc_depth_texture)};

                vkUpdateDescriptorSets(get_device().get_handle(), static_cast<uint32_t>(write_descriptor_sets.size()), write_descriptor_sets.data(), 0, nullptr);
            }
        }
    }

    // LinSSS accumulation
    {
        // Update descriptor set
        VkDescriptorImageInfo desc_out_image;
        desc_out_image.imageView   = fbos.linsss.views[0].get_handle();
        desc_out_image.sampler     = VkSampler{};
        desc_out_image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo desc_ubo_linsss = create_descriptor(*uniform_buffer_linsss_cs);

        VkDescriptorImageInfo desc_tex_W;
        desc_tex_W.imageView   = bssrdf.view_W;
        desc_tex_W.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        desc_tex_W.sampler     = bssrdf.sampler;

        VkDescriptorImageInfo desc_tex_G_ast_W;
        desc_tex_G_ast_W.imageView   = bssrdf.view_G_ast_W;
        desc_tex_G_ast_W.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        desc_tex_G_ast_W.sampler     = bssrdf.sampler;

        VkDescriptorImageInfo desc_tex_G_ast_Phi;
        desc_tex_G_ast_Phi.imageView   = G_ast_Phi_texture.view;
        desc_tex_G_ast_Phi.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        desc_tex_G_ast_Phi.sampler     = G_ast_Phi_texture.sampler;

        VkDescriptorImageInfo desc_position_texture;
        desc_position_texture.imageView   = fbos.direct_pass.views[2].get_handle();
        desc_position_texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        desc_position_texture.sampler     = fbos.direct_pass.sampler;

        VkDescriptorImageInfo desc_normal_texture;
        desc_normal_texture.imageView   = fbos.direct_pass.views[3].get_handle();
        desc_normal_texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        desc_normal_texture.sampler     = fbos.direct_pass.sampler;

        VkDescriptorImageInfo desc_depth_texture;
        desc_depth_texture.imageView   = fbos.direct_pass.views[4].get_handle();
        desc_depth_texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        desc_depth_texture.sampler     = fbos.direct_pass.sampler;

        // Descriptor set write information
        std::vector<VkWriteDescriptorSet> write_descriptor_sets =
            {
                // Binding 0 : output image
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.linsss,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    0,
                    &desc_out_image),
                // Binding 1 : uniform buffer object
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.linsss,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    1,
                    &desc_ubo_linsss),
                // Binding 2 : texture for W
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.linsss,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    2,
                    &desc_tex_W),
                // Binding 3 : texture for (G * W)
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.linsss,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    3,
                    &desc_tex_G_ast_W),
                // Binding 4 : texture for (G * Phi)
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.linsss,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    4,
                    &desc_tex_G_ast_Phi),
                // Binding 5 : texture for position
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.linsss,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    5,
                    &desc_position_texture),
                // Binding 6 : texture for normal
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.linsss,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    6,
                    &desc_normal_texture),
                // Binding 7 : texture for depth
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.linsss,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    7,
                    &desc_depth_texture)};

        vkUpdateDescriptorSets(get_device().get_handle(), static_cast<uint32_t>(write_descriptor_sets.size()), write_descriptor_sets.data(), 0, nullptr);
    }

    // Translucent shadow maps
    {
        VkDescriptorBufferInfo desc_ubo_vs     = create_descriptor(*uniform_buffer_vs);
        VkDescriptorBufferInfo desc_ubo_sss_fs = create_descriptor(*uniform_buffer_linsss_cs);
        VkDescriptorBufferInfo desc_ubo_tsm_fs = create_descriptor(*uniform_buffer_tsm_fs);

        VkDescriptorImageInfo desc_irr_texture;
        desc_irr_texture.imageView   = fbos.shadow_map.views[0].get_handle();
        desc_irr_texture.sampler     = fbos.shadow_map.sampler;
        desc_irr_texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo desc_pos_texture;
        desc_pos_texture.imageView   = fbos.shadow_map.views[1].get_handle();
        desc_pos_texture.sampler     = fbos.shadow_map.sampler;
        desc_pos_texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo desc_norm_texture;
        desc_norm_texture.imageView   = fbos.shadow_map.views[2].get_handle();
        desc_norm_texture.sampler     = fbos.shadow_map.sampler;
        desc_norm_texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo desc_bssrdf_texture;
        desc_bssrdf_texture.imageView   = bssrdf.view_W;
        desc_bssrdf_texture.sampler     = bssrdf.sampler;
        desc_bssrdf_texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Ping
        {
            VkDescriptorImageInfo desc_accum_texture;
            desc_accum_texture.imageView   = fbos.trans_sm[1].views[0].get_handle();
            desc_accum_texture.sampler     = fbos.trans_sm[1].sampler;
            desc_accum_texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            
            std::vector<VkWriteDescriptorSet> write_descriptor_sets =
            {
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.trans_sm[0],
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    0,
                    &desc_ubo_vs),
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.trans_sm[0],
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    1,
                    &desc_ubo_sss_fs),
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.trans_sm[0],
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    2,
                    &desc_ubo_tsm_fs),
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.trans_sm[0],
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    3,
                    &desc_accum_texture),
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.trans_sm[0],
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    4,
                    &desc_irr_texture),
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.trans_sm[0],
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    5,
                    &desc_pos_texture),
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.trans_sm[0],
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    6,
                    &desc_norm_texture),
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.trans_sm[0],
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    7,
                    &desc_bssrdf_texture)};
        
            vkUpdateDescriptorSets(get_device().get_handle(), static_cast<uint32_t>(write_descriptor_sets.size()), write_descriptor_sets.data(), 0, nullptr);
        }

        {
            VkDescriptorImageInfo desc_accum_texture;
            desc_accum_texture.imageView   = fbos.trans_sm[0].views[0].get_handle();
            desc_accum_texture.sampler     = fbos.trans_sm[0].sampler;
            desc_accum_texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            std::vector<VkWriteDescriptorSet> write_descriptor_sets =
            {
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.trans_sm[1],
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    0,
                    &desc_ubo_vs),
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.trans_sm[1],
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    1,
                    &desc_ubo_sss_fs),
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.trans_sm[1],
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    2,
                    &desc_ubo_tsm_fs),
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.trans_sm[1],
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    3,
                    &desc_accum_texture),
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.trans_sm[1],
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    4,
                    &desc_irr_texture),
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.trans_sm[1],
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    5,
                    &desc_pos_texture),
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.trans_sm[1],
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    6,
                    &desc_norm_texture),
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.trans_sm[1],
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    7,
                    &desc_bssrdf_texture)};
        
            vkUpdateDescriptorSets(get_device().get_handle(), static_cast<uint32_t>(write_descriptor_sets.size()), write_descriptor_sets.data(), 0, nullptr);
        }
    }

    // Deferred shading
    {
        VkDescriptorBufferInfo desc_ubo_vs = create_descriptor(*uniform_buffer_vs);
        VkDescriptorBufferInfo desc_ubo_fs = create_descriptor(*uniform_buffer_fs);

        VkDescriptorImageInfo desc_envmap_texture;
        desc_envmap_texture.imageView   = envmap_texture.view;
        desc_envmap_texture.sampler     = envmap_texture.sampler;
        desc_envmap_texture.imageLayout = envmap_texture.image_layout;

        VkDescriptorImageInfo desc_sss_texture;
        desc_sss_texture.imageView   = fbos.linsss.views[0].get_handle();
        desc_sss_texture.sampler     = fbos.linsss.sampler;
        desc_sss_texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo desc_tsm_texture;
        desc_tsm_texture.imageView   = tsm_texture.view;
        desc_tsm_texture.sampler     = tsm_texture.sampler;
        desc_tsm_texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo desc_spec_buffer;
        desc_spec_buffer.imageView   = fbos.direct_pass.views[1].get_handle();
        desc_spec_buffer.sampler     = fbos.direct_pass.sampler;
        desc_spec_buffer.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo desc_depth_buffer;
        desc_depth_buffer.imageView   = fbos.direct_pass.views[4].get_handle();
        desc_depth_buffer.sampler     = fbos.direct_pass.sampler;
        desc_depth_buffer.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Descriptor set write information
        std::vector<VkWriteDescriptorSet> write_descriptor_sets =
            {
                // Binding 0 : Vertex shader uniform buffer
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.deferred,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    0,
                    &desc_ubo_vs),
                // Binding 1 : Fragment shader uniform buffer
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.deferred,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    1,
                    &desc_ubo_fs),
                // Binding 2 : Fragment shader, envmap texture sampler
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.deferred,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    2,
                    &desc_envmap_texture),
                // Binding 3 : Fragment shader, sss texture sampler
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.deferred,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    3,
                    &desc_sss_texture),
                // Binding 4 : Fragment shader, sss texture sampler
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.deferred,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    4,
                    &desc_tsm_texture),
                // Biding 5 : Fragment shader, specular buffer sampler
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.deferred,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    5,
                    &desc_spec_buffer),
                // Biding 6 : Fragment shader, depth buffer sampler
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.deferred,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    6,
                    &desc_depth_buffer),
            };

        vkUpdateDescriptorSets(get_device().get_handle(), static_cast<uint32_t>(write_descriptor_sets.size()), write_descriptor_sets.data(), 0, nullptr);
    }

    // Postprocess
    {
        VkDescriptorBufferInfo desc_ubo_postproc_vs = create_descriptor(*uniform_buffer_postproc_vs);

        VkDescriptorImageInfo desc_source_texture;
        desc_source_texture.imageView   = fbos.deferred.views[0].get_handle();
        desc_source_texture.sampler     = fbos.deferred.sampler;
        desc_source_texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Descriptor set write information
        std::vector<VkWriteDescriptorSet> write_descriptor_sets =
            {
                // Binding 0 : Vertex shader uniform buffer
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.postprocess,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    0,
                    &desc_ubo_postproc_vs),
                // Binding 1 : Fragment shader, envmap texture sampler
                vkb::initializers::write_descriptor_set(
                    descriptor_sets.postprocess,
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    1,
                    &desc_source_texture)};

        vkUpdateDescriptorSets(get_device().get_handle(), static_cast<uint32_t>(write_descriptor_sets.size()), write_descriptor_sets.data(), 0, nullptr);
    }
}

void LinSSScatter::prepare_pipelines()
{
    // Common settings for all the pipelines
    VkPipelineInputAssemblyStateCreateInfo input_assembly_state =
        vkb::initializers::pipeline_input_assembly_state_create_info(
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            0,
            VK_FALSE);

    VkPipelineRasterizationStateCreateInfo rasterization_state =
        vkb::initializers::pipeline_rasterization_state_create_info(
            VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_NONE,
            VK_FRONT_FACE_COUNTER_CLOCKWISE,
            0);

    VkPipelineColorBlendAttachmentState blend_attachment_state =
        vkb::initializers::pipeline_color_blend_attachment_state(
            0xf,
            VK_FALSE);

    VkPipelineColorBlendStateCreateInfo color_blend_state =
        vkb::initializers::pipeline_color_blend_state_create_info(
            1,
            &blend_attachment_state);

    VkPipelineDepthStencilStateCreateInfo depth_stencil_state =
        vkb::initializers::pipeline_depth_stencil_state_create_info(
            VK_TRUE,
            VK_TRUE,
            VK_COMPARE_OP_LESS);

    VkPipelineViewportStateCreateInfo viewport_state =
        vkb::initializers::pipeline_viewport_state_create_info(1, 1, 0);

    VkPipelineMultisampleStateCreateInfo multisample_state =
        vkb::initializers::pipeline_multisample_state_create_info(
            VK_SAMPLE_COUNT_1_BIT,
            0);

    std::vector<VkDynamicState> dynamic_state_enables = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo dynamic_state =
        vkb::initializers::pipeline_dynamic_state_create_info(
            dynamic_state_enables.data(),
            static_cast<uint32_t>(dynamic_state_enables.size()),
            0);

    // Pipeline for light pass
    {
        // Multiple render targets
        std::array<VkPipelineColorBlendAttachmentState, 3> multi_blend_attachment_states = {};
        for (uint32_t i = 0; i < static_cast<uint32_t>(multi_blend_attachment_states.size()); i++)
        {
            multi_blend_attachment_states[i] = vkb::initializers::pipeline_color_blend_attachment_state(
                0xf,
                VK_FALSE);
        }
        VkPipelineColorBlendStateCreateInfo multi_color_blend_state =
            vkb::initializers::pipeline_color_blend_state_create_info(
                multi_blend_attachment_states.size(),
                multi_blend_attachment_states.data());

        // Load shaders
        std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages = {};
        shader_stages[0]                                             = load_spirv("linsss/light_pass.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        shader_stages[1]                                             = load_spirv("linsss/light_pass.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

        // Vertex bindings and attributes
        const std::vector<VkVertexInputBindingDescription> vertex_input_bindings = {
            vkb::initializers::vertex_input_binding_description(0, sizeof(LinSSScatterVertexStructure), VK_VERTEX_INPUT_RATE_VERTEX),
        };
        const std::vector<VkVertexInputAttributeDescription> vertex_input_attributes = {
            vkb::initializers::vertex_input_attribute_description(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LinSSScatterVertexStructure, pos)),
            vkb::initializers::vertex_input_attribute_description(0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(LinSSScatterVertexStructure, uv)),
            vkb::initializers::vertex_input_attribute_description(0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LinSSScatterVertexStructure, normal)),
        };
        VkPipelineVertexInputStateCreateInfo vertex_input_state = vkb::initializers::pipeline_vertex_input_state_create_info();
        vertex_input_state.vertexBindingDescriptionCount        = static_cast<uint32_t>(vertex_input_bindings.size());
        vertex_input_state.pVertexBindingDescriptions           = vertex_input_bindings.data();
        vertex_input_state.vertexAttributeDescriptionCount      = static_cast<uint32_t>(vertex_input_attributes.size());
        vertex_input_state.pVertexAttributeDescriptions         = vertex_input_attributes.data();

        VkGraphicsPipelineCreateInfo pipeline_create_info =
            vkb::initializers::pipeline_create_info(
                pipeline_layouts.light_pass,
                render_passes.light_pass,
                0);

        pipeline_create_info.pVertexInputState   = &vertex_input_state;
        pipeline_create_info.pInputAssemblyState = &input_assembly_state;
        pipeline_create_info.pRasterizationState = &rasterization_state;
        pipeline_create_info.pColorBlendState    = &multi_color_blend_state;
        pipeline_create_info.pMultisampleState   = &multisample_state;
        pipeline_create_info.pViewportState      = &viewport_state;
        pipeline_create_info.pDepthStencilState  = &depth_stencil_state;
        pipeline_create_info.pDynamicState       = &dynamic_state;
        pipeline_create_info.stageCount          = static_cast<uint32_t>(shader_stages.size());
        pipeline_create_info.pStages             = shader_stages.data();

        VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), pipeline_cache, 1, &pipeline_create_info, nullptr, &pipelines.light_pass));
    }

    // Pipeline for direct illumination
    {
        // Multiple render targets
        std::array<VkPipelineColorBlendAttachmentState, 5> multi_blend_attachment_states = {};
        for (uint32_t i = 0; i < static_cast<uint32_t>(multi_blend_attachment_states.size()); i++)
        {
            multi_blend_attachment_states[i] = vkb::initializers::pipeline_color_blend_attachment_state(
                0xf,
                VK_FALSE);
        }
        VkPipelineColorBlendStateCreateInfo multi_color_blend_state =
            vkb::initializers::pipeline_color_blend_state_create_info(
                multi_blend_attachment_states.size(),
                multi_blend_attachment_states.data());

        // Load shaders
        std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages = {};
        shader_stages[0]                                             = load_spirv("linsss/direct_pass.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        shader_stages[1]                                             = load_spirv("linsss/direct_pass.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

        // Vertex bindings and attributes
        const std::vector<VkVertexInputBindingDescription> vertex_input_bindings = {
            vkb::initializers::vertex_input_binding_description(0, sizeof(LinSSScatterVertexStructure), VK_VERTEX_INPUT_RATE_VERTEX),
        };
        const std::vector<VkVertexInputAttributeDescription> vertex_input_attributes = {
            vkb::initializers::vertex_input_attribute_description(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LinSSScatterVertexStructure, pos)),
            vkb::initializers::vertex_input_attribute_description(0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(LinSSScatterVertexStructure, uv)),
            vkb::initializers::vertex_input_attribute_description(0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LinSSScatterVertexStructure, normal)),
        };
        VkPipelineVertexInputStateCreateInfo vertex_input_state = vkb::initializers::pipeline_vertex_input_state_create_info();
        vertex_input_state.vertexBindingDescriptionCount        = static_cast<uint32_t>(vertex_input_bindings.size());
        vertex_input_state.pVertexBindingDescriptions           = vertex_input_bindings.data();
        vertex_input_state.vertexAttributeDescriptionCount      = static_cast<uint32_t>(vertex_input_attributes.size());
        vertex_input_state.pVertexAttributeDescriptions         = vertex_input_attributes.data();

        VkGraphicsPipelineCreateInfo pipeline_create_info =
            vkb::initializers::pipeline_create_info(
                pipeline_layouts.direct_pass,
                render_passes.direct_pass,
                0);

        pipeline_create_info.pVertexInputState   = &vertex_input_state;
        pipeline_create_info.pInputAssemblyState = &input_assembly_state;
        pipeline_create_info.pRasterizationState = &rasterization_state;
        pipeline_create_info.pColorBlendState    = &multi_color_blend_state;
        pipeline_create_info.pMultisampleState   = &multisample_state;
        pipeline_create_info.pViewportState      = &viewport_state;
        pipeline_create_info.pDepthStencilState  = &depth_stencil_state;
        pipeline_create_info.pDynamicState       = &dynamic_state;
        pipeline_create_info.stageCount          = static_cast<uint32_t>(shader_stages.size());
        pipeline_create_info.pStages             = shader_stages.data();

        VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), pipeline_cache, 1, &pipeline_create_info, nullptr, &pipelines.direct_pass));
    }

    // Gaussian filter
    {
        // Compute pipeline
        VkComputePipelineCreateInfo pipeline_create_info = vkb::initializers::compute_pipeline_create_info(pipeline_layouts.gauss_filter, 0);

        // Load shaders
        pipeline_create_info.stage = load_spirv("linsss/gauss_filter.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);

        // Set shader constant parameters
        struct SpecializationData
        {
            float sss_level;
            float correction;
            float maxdd;
            int   ksize;
        } specialization_data;

        std::vector<VkSpecializationMapEntry> specialization_map_entries;
        specialization_map_entries.push_back(vkb::initializers::specialization_map_entry(0, offsetof(SpecializationData, sss_level), sizeof(float)));
        specialization_map_entries.push_back(vkb::initializers::specialization_map_entry(1, offsetof(SpecializationData, correction), sizeof(float)));
        specialization_map_entries.push_back(vkb::initializers::specialization_map_entry(2, offsetof(SpecializationData, maxdd), sizeof(float)));
        specialization_map_entries.push_back(vkb::initializers::specialization_map_entry(3, offsetof(SpecializationData, ksize), sizeof(int)));

        specialization_data.sss_level  = 31.5f;
        specialization_data.correction = 800.0f;
        specialization_data.maxdd      = 0.001f;
        specialization_data.ksize      = bssrdf.ksize;

        VkSpecializationInfo specialization_info = vkb::initializers::specialization_info(static_cast<uint32_t>(specialization_map_entries.size()),
                                                                                          specialization_map_entries.data(),
                                                                                          sizeof(SpecializationData),
                                                                                          &specialization_data);

        pipeline_create_info.stage.pSpecializationInfo = &specialization_info;

        VK_CHECK(vkCreateComputePipelines(get_device().get_handle(), pipeline_cache, 1, &pipeline_create_info, nullptr, &pipelines.gauss_filter));
    }

    // LinSSS accumulation
    {
        // Compute pipeline
        VkComputePipelineCreateInfo pipeline_create_info = vkb::initializers::compute_pipeline_create_info(pipeline_layouts.linsss, 0);

        // Load shaders
        pipeline_create_info.stage = load_spirv("linsss/linsss.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);

        // Set shader constant parameters
        struct SpecializationData
        {
            int n_gauss;
        } specialization_data;

        std::vector<VkSpecializationMapEntry> specialization_map_entries;
        specialization_map_entries.push_back(vkb::initializers::specialization_map_entry(0, offsetof(SpecializationData, n_gauss), sizeof(int)));

        specialization_data.n_gauss = bssrdf.n_gauss;

        VkSpecializationInfo specialization_info = vkb::initializers::specialization_info(static_cast<uint32_t>(specialization_map_entries.size()),
                                                                                          specialization_map_entries.data(),
                                                                                          sizeof(SpecializationData),
                                                                                          &specialization_data);

        pipeline_create_info.stage.pSpecializationInfo = &specialization_info;

        VK_CHECK(vkCreateComputePipelines(get_device().get_handle(), pipeline_cache, 1, &pipeline_create_info, nullptr, &pipelines.linsss));
    }

    // Translucent shadow maps
    {
        // Load shaders
        std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages = {};
        shader_stages[0]                                             = load_spirv("linsss/translucent_shadow_maps.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        shader_stages[1]                                             = load_spirv("linsss/translucent_shadow_maps.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

        // Vertex bindings and attributes
        const std::vector<VkVertexInputBindingDescription> vertex_input_bindings = {
            vkb::initializers::vertex_input_binding_description(0, sizeof(LinSSScatterVertexStructure), VK_VERTEX_INPUT_RATE_VERTEX),
        };
        const std::vector<VkVertexInputAttributeDescription> vertex_input_attributes = {
            vkb::initializers::vertex_input_attribute_description(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LinSSScatterVertexStructure, pos)),
            vkb::initializers::vertex_input_attribute_description(0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(LinSSScatterVertexStructure, uv)),
            vkb::initializers::vertex_input_attribute_description(0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LinSSScatterVertexStructure, normal)),
        };
        VkPipelineVertexInputStateCreateInfo vertex_input_state = vkb::initializers::pipeline_vertex_input_state_create_info();
        vertex_input_state.vertexBindingDescriptionCount        = static_cast<uint32_t>(vertex_input_bindings.size());
        vertex_input_state.pVertexBindingDescriptions           = vertex_input_bindings.data();
        vertex_input_state.vertexAttributeDescriptionCount      = static_cast<uint32_t>(vertex_input_attributes.size());
        vertex_input_state.pVertexAttributeDescriptions         = vertex_input_attributes.data();

        VkGraphicsPipelineCreateInfo pipeline_create_info =
            vkb::initializers::pipeline_create_info(
                pipeline_layouts.trans_sm,
                render_passes.trans_sm,
                0);

        pipeline_create_info.pVertexInputState   = &vertex_input_state;
        pipeline_create_info.pInputAssemblyState = &input_assembly_state;
        pipeline_create_info.pRasterizationState = &rasterization_state;
        pipeline_create_info.pColorBlendState    = &color_blend_state;
        pipeline_create_info.pMultisampleState   = &multisample_state;
        pipeline_create_info.pViewportState      = &viewport_state;
        pipeline_create_info.pDepthStencilState  = &depth_stencil_state;
        pipeline_create_info.pDynamicState       = &dynamic_state;
        pipeline_create_info.stageCount          = static_cast<uint32_t>(shader_stages.size());
        pipeline_create_info.pStages             = shader_stages.data();

        VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), pipeline_cache, 1, &pipeline_create_info, nullptr, &pipelines.trans_sm));
    }

    // Pipeline for background
    {
        // Load shaders
        std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages = {};
        shader_stages[0]                                             = load_spirv("linsss/envmap.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        shader_stages[1]                                             = load_spirv("linsss/envmap.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

        // Vertex bindings and attributes
        const std::vector<VkVertexInputBindingDescription> vertex_input_bindings = {
            vkb::initializers::vertex_input_binding_description(0, sizeof(float) * 3, VK_VERTEX_INPUT_RATE_VERTEX),
        };
        const std::vector<VkVertexInputAttributeDescription> vertex_input_attributes = {
            vkb::initializers::vertex_input_attribute_description(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0),
        };
        VkPipelineVertexInputStateCreateInfo vertex_input_state = vkb::initializers::pipeline_vertex_input_state_create_info();
        vertex_input_state.vertexBindingDescriptionCount        = static_cast<uint32_t>(vertex_input_bindings.size());
        vertex_input_state.pVertexBindingDescriptions           = vertex_input_bindings.data();
        vertex_input_state.vertexAttributeDescriptionCount      = static_cast<uint32_t>(vertex_input_attributes.size());
        vertex_input_state.pVertexAttributeDescriptions         = vertex_input_attributes.data();

        VkGraphicsPipelineCreateInfo pipeline_create_info =
            vkb::initializers::pipeline_create_info(
                pipeline_layouts.deferred,
                render_passes.deferred,
                0);

        pipeline_create_info.pVertexInputState   = &vertex_input_state;
        pipeline_create_info.pInputAssemblyState = &input_assembly_state;
        pipeline_create_info.pRasterizationState = &rasterization_state;
        pipeline_create_info.pColorBlendState    = &color_blend_state;
        pipeline_create_info.pMultisampleState   = &multisample_state;
        pipeline_create_info.pViewportState      = &viewport_state;
        pipeline_create_info.pDepthStencilState  = &depth_stencil_state;
        pipeline_create_info.pDynamicState       = &dynamic_state;
        pipeline_create_info.stageCount          = static_cast<uint32_t>(shader_stages.size());
        pipeline_create_info.pStages             = shader_stages.data();

        VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), pipeline_cache, 1, &pipeline_create_info, nullptr, &pipelines.background));
    }

    // Pipeline for deferred shading
    {
        // Load shaders
        std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages = {};
        shader_stages[0]                                             = load_spirv("linsss/deferred_pass.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        shader_stages[1]                                             = load_spirv("linsss/deferred_pass.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

        // Vertex bindings and attributes
        const std::vector<VkVertexInputBindingDescription> vertex_input_bindings = {
            vkb::initializers::vertex_input_binding_description(0, sizeof(LinSSScatterVertexStructure), VK_VERTEX_INPUT_RATE_VERTEX),
        };
        const std::vector<VkVertexInputAttributeDescription> vertex_input_attributes = {
            vkb::initializers::vertex_input_attribute_description(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LinSSScatterVertexStructure, pos)),
            vkb::initializers::vertex_input_attribute_description(0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(LinSSScatterVertexStructure, uv)),
            vkb::initializers::vertex_input_attribute_description(0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LinSSScatterVertexStructure, normal)),
        };
        VkPipelineVertexInputStateCreateInfo vertex_input_state = vkb::initializers::pipeline_vertex_input_state_create_info();
        vertex_input_state.vertexBindingDescriptionCount        = static_cast<uint32_t>(vertex_input_bindings.size());
        vertex_input_state.pVertexBindingDescriptions           = vertex_input_bindings.data();
        vertex_input_state.vertexAttributeDescriptionCount      = static_cast<uint32_t>(vertex_input_attributes.size());
        vertex_input_state.pVertexAttributeDescriptions         = vertex_input_attributes.data();

        VkGraphicsPipelineCreateInfo pipeline_create_info =
            vkb::initializers::pipeline_create_info(
                pipeline_layouts.deferred,
                render_passes.deferred,
                0);

        pipeline_create_info.pVertexInputState   = &vertex_input_state;
        pipeline_create_info.pInputAssemblyState = &input_assembly_state;
        pipeline_create_info.pRasterizationState = &rasterization_state;
        pipeline_create_info.pColorBlendState    = &color_blend_state;
        pipeline_create_info.pMultisampleState   = &multisample_state;
        pipeline_create_info.pViewportState      = &viewport_state;
        pipeline_create_info.pDepthStencilState  = &depth_stencil_state;
        pipeline_create_info.pDynamicState       = &dynamic_state;
        pipeline_create_info.stageCount          = static_cast<uint32_t>(shader_stages.size());
        pipeline_create_info.pStages             = shader_stages.data();

        VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), pipeline_cache, 1, &pipeline_create_info, nullptr, &pipelines.deferred));
    }

    // Pipeline for postprocess
    {
        // Load shaders
        std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages = {};
        shader_stages[0]                                             = load_spirv("linsss/postprocess.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        shader_stages[1]                                             = load_spirv("linsss/postprocess.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

        // Vertex bindings and attributes
        const std::vector<VkVertexInputBindingDescription> vertex_input_bindings = {
            vkb::initializers::vertex_input_binding_description(0, sizeof(float) * 3, VK_VERTEX_INPUT_RATE_VERTEX),
        };
        const std::vector<VkVertexInputAttributeDescription> vertex_input_attributes = {
            vkb::initializers::vertex_input_attribute_description(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0),
        };
        VkPipelineVertexInputStateCreateInfo vertex_input_state = vkb::initializers::pipeline_vertex_input_state_create_info();
        vertex_input_state.vertexBindingDescriptionCount        = static_cast<uint32_t>(vertex_input_bindings.size());
        vertex_input_state.pVertexBindingDescriptions           = vertex_input_bindings.data();
        vertex_input_state.vertexAttributeDescriptionCount      = static_cast<uint32_t>(vertex_input_attributes.size());
        vertex_input_state.pVertexAttributeDescriptions         = vertex_input_attributes.data();

        VkGraphicsPipelineCreateInfo pipeline_create_info =
            vkb::initializers::pipeline_create_info(
                pipeline_layouts.postprocess,
                render_pass,
                0);

        pipeline_create_info.pVertexInputState   = &vertex_input_state;
        pipeline_create_info.pInputAssemblyState = &input_assembly_state;
        pipeline_create_info.pRasterizationState = &rasterization_state;
        pipeline_create_info.pColorBlendState    = &color_blend_state;
        pipeline_create_info.pMultisampleState   = &multisample_state;
        pipeline_create_info.pViewportState      = &viewport_state;
        pipeline_create_info.pDepthStencilState  = &depth_stencil_state;
        pipeline_create_info.pDynamicState       = &dynamic_state;
        pipeline_create_info.stageCount          = static_cast<uint32_t>(shader_stages.size());
        pipeline_create_info.pStages             = shader_stages.data();

        VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), pipeline_cache, 1, &pipeline_create_info, nullptr, &pipelines.postprocess));
    }
}

void LinSSScatter::prepare_uniform_buffers()
{
    // Vertex shader uniform buffer block
    uniform_buffer_sm_vs = std::make_unique<vkb::core::Buffer>(get_device(),
                                                               sizeof(ubo_sm_vs),
                                                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                               VMA_MEMORY_USAGE_CPU_TO_GPU);

    // Vertex shader uniform buffer block
    uniform_buffer_vs = std::make_unique<vkb::core::Buffer>(get_device(),
                                                            sizeof(ubo_vs),
                                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                            VMA_MEMORY_USAGE_CPU_TO_GPU);

    // Fragment shader uniform buffer block
    uniform_buffer_fs = std::make_unique<vkb::core::Buffer>(get_device(),
                                                            sizeof(ubo_fs),
                                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                            VMA_MEMORY_USAGE_CPU_TO_GPU);

    // Gaussian filter uniform buffer block
    uniform_buffer_gauss_horz_cs = std::make_unique<vkb::core::Buffer>(get_device(),
                                                                       sizeof(ubo_gauss_cs),
                                                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                                       VMA_MEMORY_USAGE_CPU_TO_GPU);

    uniform_buffer_gauss_vert_cs = std::make_unique<vkb::core::Buffer>(get_device(),
                                                                       sizeof(ubo_gauss_cs),
                                                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                                       VMA_MEMORY_USAGE_CPU_TO_GPU);

    // LinSSS accumulation
    uniform_buffer_linsss_cs = std::make_unique<vkb::core::Buffer>(get_device(),
                                                                   sizeof(ubo_linsss_cs),
                                                                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                                   VMA_MEMORY_USAGE_CPU_TO_GPU);

    // Translucent shadow maps
    uniform_buffer_tsm_fs = std::make_unique<vkb::core::Buffer>(get_device(),
                                                                sizeof(ubo_tsm_fs),
                                                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                                VMA_MEMORY_USAGE_CPU_TO_GPU);

    // Postproce3ss
    uniform_buffer_postproc_vs = std::make_unique<vkb::core::Buffer>(get_device(),
                                                                     sizeof(ubo_postproc_vs),
                                                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                                     VMA_MEMORY_USAGE_CPU_TO_GPU);

    update_uniform_buffers();
}

void LinSSScatter::update_uniform_buffers()
{
    static const glm::vec3 light_pos   = glm::vec3(5.0f, 5.0f, 0.0f);
    static const glm::vec3 light_power = glm::vec3(5.0f, 5.0f, 5.0f);
    const uint32_t         win_width   = get_render_context().get_surface_extent().width;
    const uint32_t         win_height  = get_render_context().get_surface_extent().height;

    // Shadow mapping
    {
        // Vertex shader (shadow map)
        ubo_sm_vs.projection  = glm::perspective(glm::radians(30.0f), 1.0f, 1.0f, 50.0f);
        ubo_sm_vs.model       = glm::lookAt(light_pos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        ubo_sm_vs.light_pos   = glm::vec4(light_pos, 0.0f);
        ubo_sm_vs.light_power = glm::vec4(light_power, 0.0f);

        uniform_buffer_sm_vs->convert_and_update(ubo_sm_vs);
    }

    // Gaussian filter
    {
        // It will be updated when descriptor set is updated
    }

    // LinSSS
    {
        for (int i = 0; i < bssrdf.n_gauss; i++)
        {
            ubo_linsss_cs.sigmas[i] = bssrdf.sigmas[i];
        }
        uniform_buffer_linsss_cs->convert_and_update(ubo_linsss_cs);
    }

    // Direct pass
    {
        // Vertex shader
        ubo_vs.projection     = glm::perspective(glm::radians(60.0f), (float) win_width / (float) win_height, 0.001f, 256.0f);
        glm::mat4 view_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, zoom));

        ubo_vs.model = view_matrix * glm::translate(glm::mat4(1.0f), camera_pos);
        ubo_vs.model = glm::rotate(ubo_vs.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
        ubo_vs.model = glm::rotate(ubo_vs.model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
        ubo_vs.model = glm::rotate(ubo_vs.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

        ubo_vs.sm_mvp = ubo_sm_vs.projection * ubo_sm_vs.model;

        ubo_vs.view_pos  = glm::vec4(0.0f, 0.0f, -zoom, 0.0f);
        ubo_vs.light_pos = glm::vec4(light_pos, 0.0f);

        // Fragment shader
        if (ubo_fs.light_type != LightType::Point)
        {
            std::string filename = "";
            if (ubo_fs.light_type == LightType::Uffizi)
                filename = "scenes/envmap/uffizi.sph";
            if (ubo_fs.light_type == LightType::Grace)
                filename = "scenes/envmap/grace.sph";

            // Load harmonics coefficients
            std::ifstream reader(filename.c_str(), std::ios::in);
            if (reader.fail())
            {
                LOGE("Failed to open file: {}", filename);
            }

            for (int i = 0; i < 9; i++)
            {
                glm::vec4 &c = ubo_fs.sphere_harm_coefs[i];
                reader >> c.x >> c.y >> c.z;
                c *= ENVMAP_SCALE;
            }
            reader.close();
        }
        ubo_fs.light_power = glm::vec4(light_power, 1.0);

        uniform_buffer_vs->convert_and_update(ubo_vs);
        uniform_buffer_fs->convert_and_update(ubo_fs);
    }

    // Translucent shadow maps
    {
        ubo_tsm_fs.mvp           = ubo_vs.projection * ubo_vs.model;
        ubo_tsm_fs.sm_mvp        = ubo_sm_vs.projection * ubo_sm_vs.model;
        ubo_tsm_fs.n_gauss       = bssrdf.n_gauss;
        ubo_tsm_fs.ksize         = bssrdf.ksize;
        ubo_tsm_fs.sigma_scale   = ubo_gauss_cs.sigma;
        ubo_tsm_fs.screen_extent = glm::vec2(win_width, win_height);
        ubo_tsm_fs.bssrdf_extent = glm::vec2(bssrdf.width, bssrdf.height);
        ubo_tsm_fs.seed          = glm::vec2(0.5f, 0.5f);

        uniform_buffer_tsm_fs->convert_and_update(ubo_tsm_fs);
    }

    // Postprocess
    {
        ubo_postproc_vs.win_width  = win_width;
        ubo_postproc_vs.win_height = win_height;

        uniform_buffer_postproc_vs->convert_and_update(ubo_postproc_vs);
    }
}

bool LinSSScatter::prepare(vkb::Platform &platform)
{
    if (!ApiVulkanSample::prepare(platform))
    {
        return false;
    }

    prepare_texture(envmap_texture, "scenes/envmap/uffizi.hdr", false, ENVMAP_SCALE);
    prepare_texture(Ks_texture, "scenes/bssrdf/HeartSoap_Ks.hdr", true);
    prepare_bssrdf("scenes/bssrdf/HeartSoap.sss");

    load_model("scenes/models/fertility.ply");
    prepare_primitive_objects();
    prepare_uniform_buffers();
    setup_descriptor_set_layout();
    prepare_pipelines();
    setup_descriptor_set();
    update_descriptor_set();
    build_command_buffers();

    prepared = true;
    return true;
}

void LinSSScatter::setup_render_pass()
{
    ApiVulkanSample::setup_render_pass();
    setup_custom_render_passes();
}

void LinSSScatter::setup_framebuffer()
{
    ApiVulkanSample::setup_framebuffer();
    setup_custom_framebuffers();
}

void LinSSScatter::resize(const uint32_t width, const uint32_t height)
{
    destroy_custom_framebuffers();
    ApiVulkanSample::resize(width, height);
}

void LinSSScatter::render(float delta_time)
{
    if (!prepared)
        return;
    draw();
}

void LinSSScatter::update(float delta_time)
{
    // Accumulate TSM sampling
    ubo_tsm_fs.seed = glm::vec2(frame_count);
    uniform_buffer_tsm_fs->convert_and_update(ubo_tsm_fs);
    ApiVulkanSample::update(delta_time);
}

void LinSSScatter::view_changed()
{
    // Clear TSM accumulation
    VkCommandBuffer command_buffer = get_device().create_command_buffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    VkClearColorValue clear_color{{0.0f, 0.0f, 0.0f, 1.0f}};
    VkImageSubresourceRange subresource_range;
    subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource_range.baseMipLevel = 0;
    subresource_range.levelCount = 1;
    subresource_range.baseArrayLayer = 0;
    subresource_range.layerCount = 1;

    vkCmdClearColorImage(
        command_buffer,
        fbos.trans_sm[0].images[0].get_handle(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        &clear_color,
        1,
        &subresource_range);

    vkCmdClearColorImage(
        command_buffer,
        fbos.trans_sm[1].images[0].get_handle(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        &clear_color,
        1,
        &subresource_range);

    get_device().flush_command_buffer(command_buffer, queue, true);


    // Update uniform buffers
    update_uniform_buffers();
}

void LinSSScatter::on_update_ui_overlay(vkb::Drawer &drawer)
{
    static int bssrdf_type = BSSRDFType::Heart;

    if (drawer.header("Settings"))
    {
        bool update = false;

        // Light type
        const int prev_light_type = ubo_fs.light_type;
        update |= drawer.combo_box("Light", &ubo_fs.light_type, {"Point", "Uffizi", "Grace"});

        // BSSRDF type
        const int prev_bssrdf_type = bssrdf_type;
        update |= drawer.combo_box("BSSRDF", &bssrdf_type, {"Heart", "Marble"});

        // Scaling parameters
        update |= drawer.slider_float("Irr. scale", &ubo_linsss_cs.irr_scale, 0.0f, 10.0f);
        update |= drawer.slider_float("UV scale", &ubo_linsss_cs.tex_scale, 0.5f, 2.0f);
        update |= drawer.slider_float("U offset", &ubo_linsss_cs.tex_offset_x, -1.0f, 1.0f);
        update |= drawer.slider_float("V offset", &ubo_linsss_cs.tex_offset_y, -1.0f, 1.0f);
        update |= drawer.slider_float("Sigma scale", &ubo_gauss_cs.sigma, 0.0f, 16.0f);

        // TSM        
        drawer.checkbox("TSM", &enable_tsm);

        if (update)
        {
            update_uniform_buffers();

            if (ubo_fs.light_type != LightType::Point && ubo_fs.light_type != prev_light_type)
            {
                destroy_texture(envmap_texture);
                if (ubo_fs.light_type == LightType::Uffizi)
                    prepare_texture(envmap_texture, "scenes/envmap/uffizi.hdr", false, ENVMAP_SCALE);
                if (ubo_fs.light_type == LightType::Grace)
                    prepare_texture(envmap_texture, "scenes/envmap/grace.hdr", false, ENVMAP_SCALE);
            }

            if (bssrdf_type != prev_bssrdf_type)
            {
                if (bssrdf_type == BSSRDFType::Heart)
                {
                    prepare_bssrdf("scenes/bssrdf/HeartSoap.sss");
                    prepare_texture(Ks_texture, "scenes/bssrdf/HeartSoap_Ks.hdr", false);
                }

                if (bssrdf_type == BSSRDFType::Marble)
                {
                    prepare_bssrdf("scenes/bssrdf/MarbleSoap.sss");
                    prepare_texture(Ks_texture, "scenes/bssrdf/MarbleSoap_Ks.hdr", false);
                }
            }
        }
    }
}

std::unique_ptr<vkb::Application> create_linsss()
{
    return std::make_unique<LinSSScatter>();
}
