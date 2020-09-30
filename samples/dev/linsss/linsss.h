/*
 * Copyright (c) 2020, Tatsuya Yatagawa
 * LinSSS: Linear decomposition of heterogeneous subsurface scattering for
 * real-time screen-space rendering.
 */


#pragma once

#include <ktx.h>

#include "api_vulkan_sample.h"

// Enumeration for light type
enum LightType : int
{
    Point  = 0x00,
    Uffizi = 0x01,
    Grace  = 0x02
};

// Enumeration for mesh type
enum MeshType : int
{
    Fertility = 0x00,
    Armadillo = 0x01
};

// Enumeration for BSSRDF kinds
enum BSSRDFType : int
{
    Heart  = 0x00,
    Marble = 0x01
};

// Vertex layout for this example
struct LinSSScatterVertexStructure
{
    LinSSScatterVertexStructure() = default;
    LinSSScatterVertexStructure(const glm::vec3 &pos, const glm::vec2 &uv, const glm::vec3 &normal) :
        pos(pos), uv(uv), normal(normal)
    {}

    bool operator==(const LinSSScatterVertexStructure &other) const
    {
        return pos == other.pos && uv == other.uv && normal == other.normal;
    }
    glm::vec3 pos;
    glm::vec2 uv;
    glm::vec3 normal;
};

// Hash struct for vertex layout
namespace std
{
template <>
struct hash<LinSSScatterVertexStructure>
{
    size_t operator()(const LinSSScatterVertexStructure &v) const
    {
        size_t h = 0;
        h        = hash<glm::vec3>()(v.pos) ^ (h << 1);
        h        = hash<glm::vec2>()(v.uv) ^ (h << 1);
        h        = hash<glm::vec3>()(v.normal) ^ (h << 1);
        return h;
    }
};

}        // namespace std

class LinSSScatter : public ApiVulkanSample
{
  public:
    // Graphics objects
    struct Texture
    {
        VkSampler      sampler;
        VkImage        image;
        VkImageLayout  image_layout;
        VkDeviceMemory device_memory;
        VkImageView    view;
        uint32_t       width, height;
        uint32_t       mip_levels;
    };

    struct BSSRDF
    {
        uint32_t               width, height, n_gauss, ksize;
        std::vector<glm::vec4> sigmas;

        VkImage        image_W;
        VkImageView    view_W;
        VkDeviceMemory device_memory_W;

        VkImage        image_G_ast_W;
        VkImageView    view_G_ast_W;
        VkDeviceMemory device_memory_G_ast_W;

        VkSampler sampler;
    };
    BSSRDF bssrdf;

    struct FBO
    {
        std::vector<vkb::core::Image>     images;
        std::vector<vkb::core::ImageView> views;
        VkFramebuffer                     fb;
        VkSampler                         sampler;
    };

    // Mesh models
    struct Rect
    {
        std::unique_ptr<vkb::core::Buffer> vertex_buffer;
        std::unique_ptr<vkb::core::Buffer> index_buffer;
        uint32_t                           index_count;
    } rect;

    struct Cube
    {
        std::unique_ptr<vkb::core::Buffer> vertex_buffer;
        std::unique_ptr<vkb::core::Buffer> index_buffer;
        uint32_t                           index_count;
    } cube;

    struct Model
    {
        std::unique_ptr<vkb::core::Buffer> vertex_buffer;
        std::unique_ptr<vkb::core::Buffer> index_buffer;
        uint32_t                           index_count;
    } model;

    // Uniform buffer objects
    struct
    {
        glm::mat4 projection;
        glm::mat4 model;
        glm::vec4 light_pos;
        glm::vec4 light_power;
    } ubo_sm_vs;

    struct
    {
        glm::mat4 projection;
        glm::mat4 model;
        glm::vec4 view_pos;
        glm::vec4 light_pos;
        glm::mat4 sm_mvp;
    } ubo_vs;

    struct
    {
        std::array<glm::vec4, 9> sphere_harm_coefs;
        glm::vec4                light_power;
        int                      light_type = (int) LightType::Uffizi;
    } ubo_fs;

    struct
    {
        float sigma     = 4.0f;
        int   direction = 0;
    } ubo_gauss_cs;

    struct
    {
        std::array<glm::vec4, 8> sigmas;
        float                    tex_offset_x = 0.0f;
        float                    tex_offset_y = 0.0f;
        float                    tex_scale    = 1.0f;
        float                    irr_scale    = 1.0f;
    } ubo_linsss_cs;

    struct
    {
        glm::mat4 mvp;
        glm::mat4 sm_mvp;
        glm::vec2 screen_extent;
        glm::vec2 bssrdf_extent;
        glm::vec2 seed;
        int       n_gauss;
        int       ksize;
        float     sigma_scale;
    } ubo_tsm_fs;

    struct
    {
        int win_width;
        int win_height;
    } ubo_postproc_vs;

    std::unique_ptr<vkb::core::Buffer> uniform_buffer_sm_vs;
    std::unique_ptr<vkb::core::Buffer> uniform_buffer_vs;
    std::unique_ptr<vkb::core::Buffer> uniform_buffer_fs;
    std::unique_ptr<vkb::core::Buffer> uniform_buffer_gauss_horz_cs;
    std::unique_ptr<vkb::core::Buffer> uniform_buffer_gauss_vert_cs;
    std::unique_ptr<vkb::core::Buffer> uniform_buffer_linsss_cs;
    std::unique_ptr<vkb::core::Buffer> uniform_buffer_tsm_fs;
    std::unique_ptr<vkb::core::Buffer> uniform_buffer_postproc_vs;

    // Other parameters
    bool enable_tsm = false;

    // Textures
    Texture Ks_texture;
    Texture envmap_texture;
    Texture G_ast_Phi_texture;
    Texture tsm_texture;

    // Graphics pipleline
    struct
    {
        VkPipeline light_pass;
        VkPipeline direct_pass;
        VkPipeline gauss_filter;
        VkPipeline linsss;
        VkPipeline trans_sm;
        VkPipeline background;
        VkPipeline deferred;
        VkPipeline postprocess;
    } pipelines;

    // Descriptor pools
    struct
    {
        VkDescriptorPool light_pass;
        VkDescriptorPool direct_pass;
        VkDescriptorPool gauss_filter;
        VkDescriptorPool linsss;
        VkDescriptorPool trans_sm;
        VkDescriptorPool deferred;
        VkDescriptorPool postprocess;
    } descriptor_pools;

    // Render passes
    struct
    {
        VkRenderPass light_pass;
        VkRenderPass direct_pass;
        VkRenderPass trans_sm;
        VkRenderPass deferred;
        // For postprocess "render_pass" member of ApiVulkanSample is used.
    } render_passes;

    struct
    {
        FBO shadow_map;
        FBO direct_pass;
        FBO gauss_filter_buffer;
        FBO linsss;
        FBO trans_sm[2];
        FBO deferred;
    } fbos;

    std::vector<VkImageView> in_image_mip_level_views;
    std::vector<VkImageView> out_image_mip_level_views;
    std::vector<VkImageView> buf_image_mip_level_views;

    bool enqueue_tsm_clear = true;

    struct
    {
        VkPipelineLayout light_pass;
        VkPipelineLayout direct_pass;
        VkPipelineLayout gauss_filter;
        VkPipelineLayout linsss;
        VkPipelineLayout trans_sm;
        VkPipelineLayout deferred;
        VkPipelineLayout postprocess;
    } pipeline_layouts;

    struct
    {
        VkDescriptorSet              light_pass;
        VkDescriptorSet              direct_pass;
        std::vector<VkDescriptorSet> gauss_horz_filter;
        std::vector<VkDescriptorSet> gauss_vert_filter;
        VkDescriptorSet              linsss;
        VkDescriptorSet              trans_sm[2];
        VkDescriptorSet              deferred;
        VkDescriptorSet              postprocess;
    } descriptor_sets;

    struct
    {
        VkDescriptorSetLayout light_pass;
        VkDescriptorSetLayout direct_pass;
        VkDescriptorSetLayout gauss_filter;
        VkDescriptorSetLayout linsss;
        VkDescriptorSetLayout trans_sm;
        VkDescriptorSetLayout deferred;
        VkDescriptorSetLayout postprocess;
    } descriptor_set_layouts;

    LinSSScatter();
    ~LinSSScatter();
    void         resize(const uint32_t width, const uint32_t height) override;
    virtual void request_gpu_features(vkb::PhysicalDevice &gpu) override;
    void         build_command_buffers() override;
    void         draw();
    void         load_model(const std::string &filename);

    void prepare_texture(Texture &texture, const std::string &filename, bool generateMipMap = false, float scale = 1.0f);
    void destroy_texture(Texture texture);
    void prepare_bssrdf(const std::string &filename);
    void destroy_bssrdf(BSSRDF bssrdf);

    void setup_render_pass() override;
    void setup_custom_render_passes();
    void destroy_custom_framebuffers();
    void setup_custom_framebuffers();
    void setup_framebuffer() override;
    void destroy_custom_render_passes();

    void setup_descriptor_set_layout();
    void setup_descriptor_set();
    void update_descriptor_set();
    void prepare_pipelines();
    void prepare_primitive_objects();
    void prepare_uniform_buffers();
    void update_uniform_buffers();
    bool prepare(vkb::Platform &platform) override;
    void generate_mipmap(VkCommandBuffer cmd_buffer, VkImage image, uint32_t image_width, uint32_t image_height, VkFormat format, uint32_t mip_levels);
    void gauss_filter_to_mipmap_compute(VkCommandBuffer cmd_buffer, uint32_t image_width, uint32_t image_height, uint32_t mip_levels);
    void linsss_accumulate_compute(VkCommandBuffer cmd_buffer);

    virtual void render(float delta_time) override;
    virtual void update(float delta_time) override;
    virtual void view_changed() override;
    virtual void on_update_ui_overlay(vkb::Drawer &drawer) override;

    uint32_t max_mip_levels_surface()
    {
        uint32_t w = get_render_context().get_surface_extent().width;
        uint32_t h = get_render_context().get_surface_extent().height;
        return std::ceil(std::log2(std::max(w, h)));
    }
};

std::unique_ptr<vkb::Application> create_linsss();
