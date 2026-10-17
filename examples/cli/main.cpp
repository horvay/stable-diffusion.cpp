#include <stdio.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

// #include "preprocessing.hpp"
#include "stable-diffusion.h"

#include "common/common.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"
#include "image_metadata.h"
#include "json.hpp"

namespace fs = std::filesystem;

const char* previews_str[] = {
    "none",
    "proj",
    "tae",
    "vae",
};

std::regex format_specifier_regex("(?:[^%]|^)(?:%%)*(%\\d{0,3}d)");

struct SDCliParams {
    SDMode mode             = IMG_GEN;
    std::string output_path = "output.png";
    int output_begin_idx    = -1;
    int compression_quality = 90;
    std::string image_path;
    std::string metadata_format = "text";

    sd_log_level_t log_level = SD_LOG_INFO;
    bool convert_name        = false;

    preview_t preview_method = PREVIEW_NONE;
    int preview_interval     = 1;
    std::string preview_path = "preview.png";
    int preview_fps          = 16;
    bool taesd_preview       = false;
    bool preview_noisy       = false;
    bool color               = false;
    bool metadata_raw        = false;
    bool metadata_brief      = false;
    bool metadata_all        = false;

    std::string imatrix_out;
    std::vector<std::string> imatrix_in;
    std::string multi_jobs_path;

    bool normal_exit = false;

    ArgOptions get_options() {
        ArgOptions options;

        options.string_options = {
            {"-o",
             "--output",
             "path to write result image to. you can use printf-style %d format specifiers for image sequences (default: ./output.png) (eg. output_%03d.png). Single-file video outputs support .avi, .webm, and animated .webp",
             0,
             &output_path},
            {"",
             "--image",
             "path to the image to inspect (for metadata mode)",
             0,
             &image_path},
            {"",
             "--metadata-format",
             "metadata output format, one of [text, json] (default: text)",
             0,
             &metadata_format},
            {"",
             "--preview-path",
             "path to write preview image to (default: ./preview.png). For image generation, the filename can have %03d placeholder for sequential numbering. Multi-frame previews support .avi, .webm, and animated .webp",
             0,
             &preview_path},
            {"",
             "--imat-out",
             "compute the imatrix for this run and save it to the provided path",
             0,
             &imatrix_out},
            {"",
             "--multi-jobs",
             "img_gen / vid_gen: render several jobs in one process. FILE has one job per line, a JSON array of extra "
             "arguments for that job (e.g. [\"-p\", \"...\", \"-r\", \"a.png\", \"--seed\", \"1\", \"-o\", \"out/raw.mp4\"]), "
             "appended to this command line. img_gen runs the jobs one after another with every model kept loaded. "
             "MiniMax-H3 vid_gen encodes every job's conditioning first, frees the text encoder, then samples and "
             "decodes the jobs with the diffusion weights kept loaded",
             0,
             &multi_jobs_path},
        };

        options.int_options = {
            {"",
             "--preview-interval",
             "preview interval: in each sampling pass, positive N updates every Nth denoiser step and -N previews only completed logical step N; 0 previews the final completed step of the first pass (base-resolution or high-noise). Default: 1",
             &preview_interval},
            {"",
             "--output-begin-idx",
             "starting index for output image sequence, must be non-negative (default 0 if specified %d in output path, 1 otherwise)",
             &output_begin_idx},
            {"",
             "--compression-quality",
             "compression quality of video and JPEG / WebP images (90 by default)",
             &compression_quality},
        };

        options.bool_options = {
            {"",
             "--convert-name",
             "convert tensor name (for convert mode)",
             true, &convert_name},
            {"",
             "--color",
             "colors the logging tags according to level",
             true, &color},
            {"",
             "--taesd-preview-only",
             std::string("prevents usage of taesd for decoding the final image. (for use with --preview ") + previews_str[PREVIEW_TAE] + ")",
             true, &taesd_preview},
            {"",
             "--preview-noisy",
             "enables previewing noisy inputs of the models rather than the denoised outputs",
             true, &preview_noisy},
            {"",
             "--metadata-raw",
             "include raw hex previews for unparsed metadata payloads",
             true, &metadata_raw},
            {"",
             "--metadata-brief",
             "truncate long metadata text values in text output",
             true, &metadata_brief},
            {"",
             "--metadata-all",
             "include structural/container entries such as IHDR, IDAT, and non-metadata JPEG segments",
             true, &metadata_all},

        };

        auto on_mode_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* mode_c_str = argv[index];
            if (mode_c_str != nullptr) {
                int mode_found = -1;
                for (int i = 0; i < MODE_COUNT; i++) {
                    if (!strcmp(mode_c_str, modes_str[i])) {
                        mode_found = i;
                    }
                }
                if (mode_found == -1) {
                    LOG_ERROR("error: invalid mode %s, must be one of [%s]\n",
                              mode_c_str, SD_ALL_MODES_STR);
                    exit(1);
                }
                mode = (SDMode)mode_found;
            }
            return 1;
        };

        auto on_preview_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* preview = argv[index];
            int preview_found   = -1;
            for (int m = 0; m < PREVIEW_COUNT; m++) {
                if (!strcmp(preview, previews_str[m])) {
                    preview_found = m;
                }
            }
            if (preview_found == -1) {
                LOG_ERROR("error: preview method %s", preview);
                return -1;
            }
            preview_method = (preview_t)preview_found;
            return 1;
        };

        auto on_help_arg = [&](int argc, const char** argv, int index, bool& valid) {
            normal_exit = true;
            valid       = true;
            return -1;
        };

        auto on_imatrix_in_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            imatrix_in.push_back(argv[index]);
            return 1;
        };

        options.manual_options = {
            {"-M",
             "--mode",
             "run mode, one of [img_gen, adetailer, vid_gen, upscale, convert, metadata], default: img_gen",
             on_mode_arg},
            {"",
             "--preview",
             std::string("preview method. must be one of the following [") + previews_str[0] + ", " + previews_str[1] + ", " + previews_str[2] + ", " + previews_str[3] + "] (default is " + previews_str[PREVIEW_NONE] + ")",
             on_preview_arg},
            {"-h",
             "--help",
             "show this help message and exit",
             on_help_arg},
            {"",
             "--imat-in",
             "load an imatrix file for quantization or continued collection; can be specified multiple times",
             on_imatrix_in_arg},
        };

        add_log_options(options, log_level);
        return options;
    };

    bool resolve() {
        if (mode == CONVERT) {
            if (output_path == "output.png") {
                output_path = "output.gguf";
            }
        }
        return true;
    }

    bool validate() {
        if (mode != METADATA) {
            if (output_path.length() == 0) {
                LOG_ERROR("error: the following arguments are required: output_path");
                return false;
            }
        } else {
            if (image_path.empty()) {
                LOG_ERROR("error: metadata mode needs an image path (--image)");
                return false;
            }
            if (metadata_format != "text" && metadata_format != "json") {
                LOG_ERROR("error: invalid metadata format %s, must be one of [text, json]",
                          metadata_format.c_str());
                return false;
            }
        }
        return true;
    }

    bool resolve_and_validate() {
        if (!resolve()) {
            return false;
        }
        if (!validate()) {
            return false;
        }
        return true;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "SDCliParams {\n"
            << "  mode: " << modes_str[mode] << ",\n"
            << "  output_path: \"" << output_path << "\",\n"
            << "  image_path: \"" << image_path << "\",\n"
            << "  metadata_format: \"" << metadata_format << "\",\n"
            << "  log_level: " << log_level_name(log_level) << ",\n"
            << "  color: " << (color ? "true" : "false") << ",\n"
            << "  convert_name: " << (convert_name ? "true" : "false") << ",\n"
            << "  preview_method: " << previews_str[preview_method] << ",\n"
            << "  preview_interval: " << preview_interval << ",\n"
            << "  preview_path: \"" << preview_path << "\",\n"
            << "  preview_fps: " << preview_fps << ",\n"
            << "  taesd_preview: " << (taesd_preview ? "true" : "false") << ",\n"
            << "  preview_noisy: " << (preview_noisy ? "true" : "false") << ",\n"
            << "  imatrix_out: \"" << imatrix_out << "\",\n"
            << "  metadata_raw: " << (metadata_raw ? "true" : "false") << ",\n"
            << "  metadata_brief: " << (metadata_brief ? "true" : "false") << ",\n"
            << "  metadata_all: " << (metadata_all ? "true" : "false") << "\n"
            << "}";
        return oss.str();
    }
};

void print_usage(int argc, const char* argv[], const std::vector<ArgOptions>& options_list) {
    std::cout << version_string() << "\n";
    std::cout << "Usage: " << argv[0] << " [options]\n\n";
    std::cout << "CLI Options:\n";
    options_list[0].print();
    std::cout << "\nContext Options:\n";
    options_list[1].print();
    std::cout << "\nGeneration Options:\n";
    options_list[2].print();
}

void parse_args(int argc, const char** argv, SDCliParams& cli_params, SDContextParams& ctx_params, SDGenerationParams& gen_params) {
    std::vector<ArgOptions> options_vec = {cli_params.get_options(), ctx_params.get_options(), gen_params.get_options()};

    if (!parse_options(argc, argv, options_vec)) {
        print_usage(argc, argv, options_vec);
        exit(cli_params.normal_exit ? 0 : 1);
    }

    log_level = cli_params.log_level;
    log_color = cli_params.color;

    bool valid = cli_params.resolve_and_validate();
    if (valid && cli_params.mode != METADATA) {
        valid = ctx_params.resolve_and_validate(cli_params.mode) &&
                gen_params.resolve_and_validate(cli_params.mode,
                                                ctx_params.lora_model_dir,
                                                ctx_params.hires_upscalers_dir);
    }

    if (!valid) {
        print_usage(argc, argv, options_vec);
        exit(1);
    }
}

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDCliParams* cli_params = (SDCliParams*)data;
    log_print(level, log, cli_params->log_level, cli_params->color);
}

bool load_images_from_dir(const std::string dir,
                          std::vector<SDImageOwner>& images,
                          int max_image_num = 0) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        LOG_ERROR("'%s' is not a valid directory\n", dir.c_str());
        return false;
    }

    std::vector<fs::directory_entry> entries;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            entries.push_back(entry);
        }
    }

    std::sort(entries.begin(), entries.end(),
              [](const fs::directory_entry& a, const fs::directory_entry& b) {
                  return a.path().filename().string() < b.path().filename().string();
              });

    for (const auto& entry : entries) {
        std::string path = entry.path().string();
        std::string ext  = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp") {
            LOG_VERBOSE("load image %zu from '%s'", images.size(), path.c_str());
            int width             = 0;
            int height            = 0;
            int loaded_channel    = 0;
            uint8_t* image_buffer = load_image_from_file(path.c_str(), width, height, loaded_channel, 0, 0);
            if (image_buffer == nullptr) {
                LOG_ERROR("load image from '%s' failed", path.c_str());
                return false;
            }

            images.emplace_back(sd_image_t{(uint32_t)width,
                                           (uint32_t)height,
                                           (uint32_t)loaded_channel,
                                           image_buffer});

            if (max_image_num > 0 && static_cast<int>(images.size()) >= max_image_num) {
                break;
            }
        }
    }
    return true;
}

std::string format_frame_idx(std::string pattern, int frame_idx) {
    std::smatch match;
    std::string result = pattern;
    while (std::regex_search(result, match, format_specifier_regex)) {
        std::string specifier = match.str(1);
        char buffer[32];
        snprintf(buffer, sizeof(buffer), specifier.c_str(), frame_idx);
        result.replace(match.position(1), match.length(1), buffer);
    }

    // Then replace all '%%' with '%'
    size_t pos = 0;
    while ((pos = result.find("%%", pos)) != std::string::npos) {
        result.replace(pos, 2, "%");
        pos += 1;
    }
    return result;
}

int continuous_preview_counter = 0;

void step_callback(int step, int frame_count, sd_image_t* image, bool is_noisy, void* data) {
    (void)step;
    (void)is_noisy;
    SDCliParams* cli_params = (SDCliParams*)data;
    // is_noisy is set to true if the preview corresponds to noisy latents, false if it's denoised latents
    // unused in this app, it will either be always noisy or always denoised here
    if (frame_count == 1) {
        fs::path path = cli_params->preview_path;
        if (encoded_image_format_from_path(path.string()) == EncodedImageFormat::UNKNOWN)
            path += ".png";
        if (std::regex_search(path.string(), format_specifier_regex))
            path = fs::path(format_frame_idx(path.string(), continuous_preview_counter++));
        if (!write_image_to_file(path.string(),
                                 image->data,
                                 image->width,
                                 image->height,
                                 image->channel,
                                 "",
                                 cli_params->compression_quality)) {
            LOG_ERROR("save preview image to '%s' failed", path.string().c_str());
        }
    } else {
        int fps = cli_params->preview_method == PREVIEW_PROJ ? cli_params->preview_fps / 4 : cli_params->preview_fps;
        if (create_video_from_sd_images(cli_params->preview_path.c_str(), image, frame_count, fps, cli_params->compression_quality) != 0) {
            LOG_ERROR("save preview video to '%s' failed", cli_params->preview_path.c_str());
        }
    }
}

static fs::path get_video_audio_sidecar_path(const SDCliParams& cli_params) {
    fs::path out_path     = cli_params.output_path;
    fs::path base_path    = out_path;
    fs::path ext          = out_path.has_extension() ? out_path.extension() : fs::path{};
    std::string ext_lower = ext.string();
    std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
    const EncodedImageFormat output_format = encoded_image_format_from_path(out_path.string());
    if (!ext.empty()) {
        if (output_format == EncodedImageFormat::JPEG ||
            output_format == EncodedImageFormat::PNG ||
            output_format == EncodedImageFormat::WEBP ||
            ext_lower == ".avi" ||
            ext_lower == ".webm") {
            base_path.replace_extension();
        }
    }
    base_path += ".wav";
    return base_path;
}

bool save_results(const SDCliParams& cli_params,
                  const SDContextParams& ctx_params,
                  const SDGenerationParams& gen_params,
                  sd_image_t* results,
                  int num_results,
                  const sd_audio_t* generated_audio = nullptr) {
    if (results == nullptr || num_results <= 0) {
        return false;
    }

    namespace fs      = std::filesystem;
    fs::path out_path = cli_params.output_path;

    if (!out_path.parent_path().empty()) {
        std::error_code ec;
        fs::create_directories(out_path.parent_path(), ec);
        if (ec) {
            LOG_ERROR("failed to create directory '%s': %s",
                      out_path.parent_path().string().c_str(), ec.message().c_str());
            return false;
        }
    }

    fs::path base_path = out_path;
    fs::path ext       = out_path.has_extension() ? out_path.extension() : fs::path{};

    std::string ext_lower = ext.string();
    std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
    const EncodedImageFormat output_format = encoded_image_format_from_path(out_path.string());
    if (!ext.empty()) {
        if (output_format == EncodedImageFormat::JPEG ||
            output_format == EncodedImageFormat::PNG ||
            output_format == EncodedImageFormat::WEBP ||
            ext_lower == ".avi" ||
            ext_lower == ".webm") {
            base_path.replace_extension();
        }
    }

    int output_begin_idx = cli_params.output_begin_idx;
    if (output_begin_idx < 0) {
        output_begin_idx = 0;
    }

    auto write_image = [&](const fs::path& path, int idx) {
        const sd_image_t& img = results[idx];
        if (!img.data)
            return false;

        int images_per_batch        = gen_params.batch_count > 0 ? std::max(1, num_results / gen_params.batch_count) : 1;
        const int64_t metadata_seed = cli_params.mode == VID_GEN ? gen_params.seed : gen_params.seed + idx / images_per_batch;
        std::string params          = gen_params.embed_image_metadata
                                          ? get_image_params(ctx_params, gen_params, metadata_seed, cli_params.mode)
                                          : "";
        const bool ok               = write_image_to_file(path.string(), img.data, img.width, img.height, img.channel, params, cli_params.compression_quality);
        LOG_INFO("save result image %d to '%s' (%s)", idx, path.string().c_str(), ok ? "success" : "failure");
        return ok;
    };

    auto write_audio_sidecar = [&](const fs::path& wav_path) {
        if (generated_audio == nullptr) {
            return;
        }
        if (write_wav_to_file(wav_path.string(),
                              generated_audio->data,
                              generated_audio->sample_count,
                              generated_audio->channels,
                              generated_audio->sample_rate)) {
            LOG_INFO("save result audio to '%s'", wav_path.string().c_str());
        } else {
            LOG_WARN("failed to save result audio to '%s'", wav_path.string().c_str());
        }
    };

    int sucessful_reults = 0;

    if (std::regex_search(cli_params.output_path, format_specifier_regex)) {
        if (output_format == EncodedImageFormat::UNKNOWN)
            ext = ".png";
        fs::path pattern = base_path;
        pattern += ext;

        for (int i = 0; i < num_results; ++i) {
            fs::path img_path = format_frame_idx(pattern.string(), output_begin_idx + i);
            if (write_image(img_path, i)) {
                sucessful_reults++;
            }
        }
        LOG_INFO("%d/%d images saved", sucessful_reults, num_results);
        return sucessful_reults != 0;
    }

    if (cli_params.mode == VID_GEN && num_results > 1) {
        if (ext_lower != ".avi" && ext_lower != ".webp" && ext_lower != ".webm")
            ext = ".avi";
        std::string params = gen_params.embed_image_metadata
                                 ? get_image_params(ctx_params, gen_params, gen_params.seed, cli_params.mode)
                                 : "";

        fs::path video_path = base_path;
        video_path += ext;
        std::string final_ext_lower = ext.string();
        std::transform(final_ext_lower.begin(), final_ext_lower.end(), final_ext_lower.begin(), ::tolower);
        const bool mux_audio = generated_audio != nullptr && (final_ext_lower == ".avi" || final_ext_lower == ".webm");
        if (create_video_from_sd_images(video_path.string().c_str(), results, num_results, gen_params.fps, cli_params.compression_quality, mux_audio ? generated_audio : nullptr, params) == 0) {
            LOG_INFO("save result video to '%s'", video_path.string().c_str());
            if (generated_audio != nullptr && !mux_audio) {
                fs::path wav_path = video_path;
                wav_path.replace_extension(".wav");
                write_audio_sidecar(wav_path);
            }
            return true;
        } else {
            LOG_ERROR("Failed to save result video to '%s'", video_path.string().c_str());
            return false;
        }
    }

    if (output_format == EncodedImageFormat::UNKNOWN)
        ext = ".png";

    for (int i = 0; i < num_results; ++i) {
        fs::path img_path = base_path;
        if (num_results > 1) {
            img_path += "_" + std::to_string(output_begin_idx + i);
        }
        img_path += ext;
        if (write_image(img_path, i)) {
            sucessful_reults++;
        }
    }
    LOG_INFO("%d/%d images saved", sucessful_reults, num_results);
    if (generated_audio != nullptr) {
        write_audio_sidecar(get_video_audio_sidecar_path(cli_params));
    }
    return sucessful_reults != 0;
}

static bool apply_adetailer(sd_ctx_t* sd_ctx,
                            const sd_ctx_params_t& sd_ctx_params,
                            const SDContextParams& ctx_params,
                            const SDGenerationParams& gen_params,
                            const sd_img_gen_params_t& img_gen_params,
                            SDMode mode,
                            SDImageVec& results,
                            int num_results) {
    if (gen_params.ad_model_path.empty()) {
        return true;
    }

    sd_adetailer_params_t ad_params{};
    ad_params.prompt          = gen_params.ad_prompt.empty() ? nullptr : gen_params.ad_prompt.c_str();
    ad_params.negative_prompt = gen_params.ad_negative_prompt.empty() ? nullptr : gen_params.ad_negative_prompt.c_str();
    ad_params.extra_ad_args   = gen_params.extra_ad_args.c_str();

    ADetailerCtxPtr ad_ctx(new_adetailer_ctx(gen_params.ad_model_path.c_str(),
                                             ctx_params.n_threads,
                                             sd_ctx_params.backend,
                                             sd_ctx_params.params_backend));
    if (ad_ctx == nullptr) {
        LOG_ERROR("new_adetailer_ctx failed");
        return false;
    }

    for (int i = 0; i < num_results; ++i) {
        if (results[i].data == nullptr) {
            continue;
        }
        sd_img_gen_params_t ad_generation_params = img_gen_params;
        ad_generation_params.seed                = img_gen_params.seed + i;
        if (mode == IMG_GEN) {
            ad_generation_params.width    = 512;
            ad_generation_params.height   = 512;
            ad_generation_params.strength = 0.4f;
        }
        sd_image_t* detailed_images = nullptr;
        int detailed_count          = 0;
        if (!adetail_image(ad_ctx.get(),
                           sd_ctx,
                           results[i],
                           &ad_params,
                           &ad_generation_params,
                           &detailed_images,
                           &detailed_count) ||
            detailed_count <= 0 || detailed_images == nullptr || detailed_images[0].data == nullptr) {
            free_sd_images(detailed_images, detailed_count);
            LOG_ERROR("ADetailer failed for image %d", i + 1);
            return false;
        }
        free(results[i].data);
        results[i]         = detailed_images[0];
        detailed_images[0] = {0, 0, 0, nullptr};
        free_sd_images(detailed_images, detailed_count);
    }
    return true;
}

// --upscale-model / --upscale-repeats: upscale every result in place (main and the --multi-jobs image path).
static void apply_upscaler(const SDContextParams& ctx_params,
                           const SDGenerationParams& gen_params,
                           const sd_ctx_params_t& sd_ctx_params,
                           SDImageVec& results,
                           int num_results) {
    int upscale_factor = 4;  // unused for RealESRGAN_x4plus_anime_6B.pth
    if (ctx_params.esrgan_path.size() > 0 && gen_params.upscale_repeats > 0) {
        UpscalerCtxPtr upscaler_ctx(new_upscaler_ctx(ctx_params.esrgan_path.c_str(),
                                                     ctx_params.diffusion_conv_direct,
                                                     ctx_params.n_threads,
                                                     gen_params.upscale_tile_size,
                                                     sd_ctx_params.backend,
                                                     sd_ctx_params.params_backend));

        if (upscaler_ctx == nullptr) {
            LOG_ERROR("new_upscaler_ctx failed");
        } else {
            for (int i = 0; i < num_results; i++) {
                if (results[i].data == nullptr) {
                    continue;
                }
                SDImageOwner current_image(results[i]);
                results[i] = {0, 0, 0, nullptr};
                for (int u = 0; u < gen_params.upscale_repeats; ++u) {
                    sd_image_t* upscaled_images = nullptr;
                    int upscaled_count          = 0;
                    bool upscale_ok             = upscale(upscaler_ctx.get(),
                                                          current_image.get(),
                                                          upscale_factor,
                                                          &upscaled_images,
                                                          &upscaled_count);
                    if (!upscale_ok || upscaled_count <= 0 || upscaled_images[0].data == nullptr) {
                        free_sd_images(upscaled_images, upscaled_count);
                        LOG_ERROR("upscale failed");
                        break;
                    }
                    sd_image_t upscaled_image = upscaled_images[0];
                    upscaled_images[0]        = {0, 0, 0, nullptr};
                    free_sd_images(upscaled_images, upscaled_count);
                    current_image.reset(upscaled_image);
                }
                results[i] = current_image.release();  // Set the final upscaled image as the result
            }
        }
    }
}

// Load every image / audio / frame-directory input named in gen_params (main and the --multi-jobs path).
static bool load_generation_inputs(SDMode mode, SDGenerationParams& gen_params) {
    auto load_image_and_update_size = [&](const std::string& path,
                                          SDImageOwner& image,
                                          int expected_channel = 3) -> bool {
        if (!load_sd_image_from_file(image.put(), path.c_str(), 0, 0, expected_channel)) {
            LOG_ERROR("load image from '%s' failed", path.c_str());
            return false;
        }

        gen_params.set_width_and_height_if_unset(image.get().width, image.get().height);
        return true;
    };

    auto load_audio = [&](const std::string& path, SDAudioOwner& audio) -> bool {
        std::vector<float> samples;
        uint32_t sample_rate = 0;
        uint32_t channels    = 0;
        if (!load_wav_from_file(path, samples, sample_rate, channels)) {
            LOG_ERROR("load WAV audio from '%s' failed", path.c_str());
            return false;
        }
        audio.reset(std::move(samples), sample_rate, channels);
        return true;
    };

    if (gen_params.init_image_path.size() > 0) {
        const bool native_init = mode == IMG_GEN || mode == ADETAILER;
        if (!load_image_and_update_size(gen_params.init_image_path, gen_params.init_image, native_init ? 0 : 3)) {
            return false;
        }
    }

    if (gen_params.end_image_path.size() > 0) {
        if (!load_image_and_update_size(gen_params.end_image_path, gen_params.end_image)) {
            return false;
        }
    }

    if (gen_params.ref_image_paths.size() > 0) {
        gen_params.ref_images.clear();
        for (auto& path : gen_params.ref_image_paths) {
            SDImageOwner ref_image({0, 0, 0, nullptr});
            if (!load_image_and_update_size(path, ref_image, 0)) {
                return false;
            }
            gen_params.ref_images.push_back(std::move(ref_image));
        }
    }

    if (!gen_params.ref_video_paths.empty()) {
        gen_params.ref_videos.clear();
        gen_params.ref_videos.reserve(gen_params.ref_video_paths.size());
        for (const auto& path : gen_params.ref_video_paths) {
            std::vector<SDImageOwner> frames;
            if (!load_images_from_dir(path, frames) || frames.empty()) {
                LOG_ERROR("load reference video frames from '%s' failed", path.c_str());
                return false;
            }
            gen_params.ref_videos.push_back(std::move(frames));
        }

        gen_params.ref_video_audios.clear();
        gen_params.ref_video_audios.resize(gen_params.ref_videos.size());
        for (size_t i = 0; i < gen_params.ref_video_audio_paths.size(); ++i) {
            if (!load_audio(gen_params.ref_video_audio_paths[i], gen_params.ref_video_audios[i])) {
                return false;
            }
        }
    }

    if (!gen_params.ref_audio_paths.empty()) {
        gen_params.ref_audios.clear();
        gen_params.ref_audios.resize(gen_params.ref_audio_paths.size());
        for (size_t i = 0; i < gen_params.ref_audio_paths.size(); ++i) {
            if (!load_audio(gen_params.ref_audio_paths[i], gen_params.ref_audios[i])) {
                return false;
            }
        }
    }

    auto split_spec = [](const std::string& spec, std::string& path, int& frame, float& strength) -> bool {
        std::vector<std::string> parts;
        size_t at = spec.find('@');
        path      = spec.substr(0, at);
        while (at != std::string::npos) {
            size_t start = at + 1;
            at           = spec.find('@', start);
            parts.push_back(spec.substr(start, at == std::string::npos ? std::string::npos : at - start));
        }
        if (path.empty() || parts.empty() || parts.size() > 2) {
            return false;
        }
        try {
            frame    = std::stoi(parts[0]);
            strength = parts.size() > 1 ? std::stof(parts[1]) : 1.f;
        } catch (...) {
            return false;
        }
        return true;
    };

    gen_params.guide_images.clear();
    gen_params.guide_videos.clear();
    gen_params.guide_frame_idxs.clear();
    gen_params.guide_strengths.clear();
    for (const auto& spec : gen_params.guide_specs) {
        std::string path;
        int frame      = 0;
        float strength = 1.f;
        if (!split_spec(spec, path, frame, strength)) {
            LOG_ERROR("invalid --guide '%s', expected PATH@FRAME[@STRENGTH]", spec.c_str());
            return false;
        }
        SDImageOwner image({0, 0, 0, nullptr});
        std::vector<SDImageOwner> frames;
        if (fs::is_directory(path)) {
            // video guide: every image in the directory, in name order, from FRAME on
            if (!load_images_from_dir(path, frames, gen_params.video_frames) || frames.empty()) {
                LOG_ERROR("--guide '%s': no frames in the directory", spec.c_str());
                return false;
            }
        } else if (!load_image_and_update_size(path, image)) {
            return false;
        }
        gen_params.guide_images.push_back(std::move(image));
        gen_params.guide_videos.push_back(std::move(frames));
        gen_params.guide_frame_idxs.push_back(frame);
        gen_params.guide_strengths.push_back(strength);
    }

    gen_params.guide_audios.clear();
    gen_params.guide_audio_frame_idxs.clear();
    gen_params.guide_audios.resize(gen_params.guide_audio_specs.size());
    for (size_t i = 0; i < gen_params.guide_audio_specs.size(); ++i) {
        std::string path;
        int frame      = 0;
        float strength = 1.f;
        if (!split_spec(gen_params.guide_audio_specs[i], path, frame, strength) ||
            !load_audio(path, gen_params.guide_audios[i])) {
            LOG_ERROR("invalid --guide-audio '%s', expected PATH@FRAME", gen_params.guide_audio_specs[i].c_str());
            return false;
        }
        gen_params.guide_audio_frame_idxs.push_back(frame);
    }

    if (gen_params.mask_image_path.size() > 0) {
        if (!load_sd_image_from_file(gen_params.mask_image.put(),
                                     gen_params.mask_image_path.c_str(),
                                     0,
                                     0,
                                     1)) {
            LOG_ERROR("load image from '%s' failed", gen_params.mask_image_path.c_str());
            return false;
        }
    }

    if (gen_params.control_image_path.size() > 0) {
        if (!load_sd_image_from_file(gen_params.control_image.put(),
                                     gen_params.control_image_path.c_str(),
                                     0,
                                     0)) {
            LOG_ERROR("load image from '%s' failed", gen_params.control_image_path.c_str());
            return false;
        }
    }

    if (gen_params.ip_adapter_image_path.size() > 0) {
        if (!load_sd_image_from_file(gen_params.ip_adapter_image.put(),
                                     gen_params.ip_adapter_image_path.c_str(),
                                     0,
                                     0)) {
            LOG_ERROR("load image from '%s' failed", gen_params.ip_adapter_image_path.c_str());
            return false;
        }
    }

    if (!gen_params.control_video_path.empty()) {
        gen_params.control_frames.clear();
        if (!load_images_from_dir(gen_params.control_video_path,
                                  gen_params.control_frames,
                                  gen_params.video_frames)) {
            return false;
        }
    }

    if (!gen_params.pm_id_images_dir.empty()) {
        gen_params.pm_id_images.clear();
        if (!load_images_from_dir(gen_params.pm_id_images_dir,
                                  gen_params.pm_id_images)) {
            return false;
        }
    }

    return true;
}

// --multi-jobs FILE: each line is a JSON array of extra arguments for one job, parsed together with this command
// line exactly as a separate sd-cli run would parse them. vid_gen jobs run through generate_videos(); img_gen jobs
// run one by one through generate_image() in one context (run_multi_image_jobs).
struct MultiJob {
    SDCliParams cli;
    SDContextParams ctx;
    SDGenerationParams gen;
    sd_vid_gen_params_t vid{};
};

struct MultiJobState {
    std::vector<std::unique_ptr<MultiJob>>* jobs = nullptr;
    int saved                                    = 0;
};

static void multi_job_result(int index, sd_image_t* frames, int num_frames, sd_audio_t* audio, int fps, void* data) {
    auto* state   = static_cast<MultiJobState*>(data);
    MultiJob& job = *(*state->jobs)[static_cast<size_t>(index)];
    if (frames == nullptr || num_frames <= 0) {
        LOG_ERROR("[multi] job %d (%s): generation failed", index + 1, job.cli.output_path.c_str());
        free_sd_audio(audio);
        return;
    }
    if (fps > 0) {
        job.gen.fps = fps;
    }
    if (save_results(job.cli, job.ctx, job.gen, frames, num_frames, audio)) {
        ++state->saved;
        LOG_INFO("[multi] job %d: saved %s", index + 1, job.cli.output_path.c_str());
    } else {
        LOG_ERROR("[multi] job %d: saving %s failed", index + 1, job.cli.output_path.c_str());
    }
    free_sd_images(frames, num_frames);
    free_sd_audio(audio);
}

// --multi-jobs with -M img_gen: one context (diffusion model, text encoder, VAE and any SD_XPU_QWEN21_LIB block
// weights stay loaded), then for each job exactly what a single run does after new_sd_ctx(): load its inputs,
// generate_image(), ADetailer, upscaler, save. A job's inputs, results and parsed parameters are freed as soon as it
// is saved; generate_image() releases its own per-run compute buffers. A failed job is logged and skipped.
static int64_t multi_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int run_multi_image_jobs(std::vector<std::unique_ptr<MultiJob>>& jobs,
                                const SDCliParams& base_cli,
                                SDContextParams& base_ctx,
                                bool cache_size_given) {
    // Jobs that share a negative prompt (and reference images) reuse its encoding; a cache hit returns the stored
    // result of the identical computation. --conditioning-cache-size on the command line overrides.
    if (!cache_size_given && base_ctx.conditioning_cache_size == 0) {
        base_ctx.conditioning_cache_size = 4;
    }
    sd_ctx_params_t sd_ctx_params = base_ctx.to_sd_ctx_params_t(base_cli.taesd_preview);
    const int64_t load_start      = multi_now_ms();
    SDCtxPtr sd_ctx(new_sd_ctx(&sd_ctx_params));
    if (sd_ctx == nullptr) {
        LOG_INFO("new_sd_ctx_t failed");
        return 1;
    }
    LOG_INFO("[multi] context ready in %.2fs; %zu image jobs", (multi_now_ms() - load_start) / 1000.f, jobs.size());

    const size_t total = jobs.size();
    int saved          = 0;
    const int64_t t0   = multi_now_ms();
    for (size_t i = 0; i < total; ++i) {
        std::unique_ptr<MultiJob> job = std::move(jobs[i]);
        const std::string out_path    = job->cli.output_path;
        const int64_t job_start       = multi_now_ms();
        LOG_INFO("[multi] job %zu/%zu start: %s (seed %lld)", i + 1, total, out_path.c_str(), (long long)job->gen.seed);
        bool ok                 = false;
        SDGenerationParams& gen = job->gen;
        if (!load_generation_inputs(job->cli.mode, gen)) {
            LOG_ERROR("[multi] job %zu: loading the inputs failed", i + 1);
        } else {
            if (gen.sample_params.sample_method == SAMPLE_METHOD_COUNT) {
                gen.sample_params.sample_method = sd_get_default_sample_method(sd_ctx.get());
            }
            if (gen.high_noise_sample_params.sample_method == SAMPLE_METHOD_COUNT) {
                gen.high_noise_sample_params.sample_method = sd_get_default_sample_method(sd_ctx.get());
            }
            if (gen.sample_params.scheduler == SCHEDULER_COUNT) {
                gen.sample_params.scheduler = sd_get_default_scheduler(sd_ctx.get(), gen.sample_params.sample_method);
            }
            sd_img_gen_params_t img_gen_params = gen.to_sd_img_gen_params_t();
            SDImageVec results;
            int num_results              = 0;
            sd_image_t* generated_images = nullptr;
            if (!generate_image(sd_ctx.get(), &img_gen_params, &generated_images, &num_results)) {
                generated_images = nullptr;
                num_results      = 0;
            }
            results.adopt(generated_images, num_results);
            if (!results) {
                LOG_ERROR("[multi] job %zu: generate failed", i + 1);
            } else if (apply_adetailer(sd_ctx.get(), sd_ctx_params, job->ctx, gen, img_gen_params, job->cli.mode, results, num_results)) {
                apply_upscaler(base_ctx, gen, sd_ctx_params, results, num_results);
                ok = save_results(job->cli, job->ctx, gen, results.data(), num_results);
            }
        }
        job.reset();  // the job's input images, parameters and results
        if (ok) {
            ++saved;
            LOG_INFO("[multi] job %zu/%zu saved %s in %.2fs", i + 1, total, out_path.c_str(), (multi_now_ms() - job_start) / 1000.f);
        } else {
            LOG_ERROR("[multi] job %zu/%zu (%s) failed after %.2fs", i + 1, total, out_path.c_str(), (multi_now_ms() - job_start) / 1000.f);
        }
    }
    LOG_INFO("[multi] %d/%zu jobs saved in %.2fs (plus %.2fs context load)", saved, total, (multi_now_ms() - t0) / 1000.f,
             (t0 - load_start) / 1000.f);
    return saved == static_cast<int>(total) ? 0 : 1;
}

static int run_multi_jobs(int argc, const char* argv[], const SDCliParams& base_cli, SDContextParams& base_ctx) {
    if (base_cli.mode != VID_GEN && base_cli.mode != IMG_GEN) {
        LOG_ERROR("--multi-jobs needs -M img_gen or -M vid_gen");
        return 1;
    }
    std::vector<std::string> base_args;
    bool cache_size_given = false;
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--multi-jobs") {
            ++i;  // and its value
            continue;
        }
        if (std::string(argv[i]) == "--conditioning-cache-size") {
            cache_size_given = true;
        }
        base_args.push_back(argv[i]);
    }
    std::ifstream file(base_cli.multi_jobs_path);
    if (!file) {
        LOG_ERROR("cannot open --multi-jobs file '%s'", base_cli.multi_jobs_path.c_str());
        return 1;
    }
    std::vector<std::unique_ptr<MultiJob>> jobs;
    std::string line;
    int line_no = 0;
    while (std::getline(file, line)) {
        ++line_no;
        const size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        std::vector<std::string> args = base_args;
        try {
            auto parsed = nlohmann::json::parse(line);
            if (!parsed.is_array()) {
                throw std::runtime_error("not a JSON array");
            }
            for (const auto& item : parsed) {
                args.push_back(item.get<std::string>());
            }
        } catch (const std::exception& e) {
            LOG_ERROR("--multi-jobs line %d: %s", line_no, e.what());
            return 1;
        }
        std::vector<const char*> job_argv;
        for (const auto& a : args) {
            job_argv.push_back(a.c_str());
        }
        auto job = std::make_unique<MultiJob>();
        std::vector<ArgOptions> options = {job->cli.get_options(), job->ctx.get_options(), job->gen.get_options()};
        if (!parse_options(static_cast<int>(job_argv.size()), job_argv.data(), options) ||
            !job->cli.resolve_and_validate() ||
            !job->ctx.resolve_and_validate(job->cli.mode) ||
            !job->gen.resolve_and_validate(job->cli.mode, job->ctx.lora_model_dir, job->ctx.hires_upscalers_dir)) {
            LOG_ERROR("--multi-jobs line %d: invalid arguments", line_no);
            return 1;
        }
        if (!job->cli.multi_jobs_path.empty() || job->cli.mode != base_cli.mode) {
            LOG_ERROR("--multi-jobs line %d: jobs cannot change the mode or nest --multi-jobs", line_no);
            return 1;
        }
        if (base_cli.mode == IMG_GEN) {
            // Inputs are loaded right before each job runs (and freed after it); check that they exist now.
            std::vector<std::string> inputs = job->gen.ref_image_paths;
            for (const std::string* path : {&job->gen.init_image_path, &job->gen.mask_image_path,
                                            &job->gen.control_image_path, &job->gen.ip_adapter_image_path}) {
                if (!path->empty()) {
                    inputs.push_back(*path);
                }
            }
            for (const auto& path : inputs) {
                if (!fs::exists(path)) {
                    LOG_ERROR("--multi-jobs line %d: input '%s' not found", line_no, path.c_str());
                    return 1;
                }
            }
            LOG_INFO("[multi] queued #%zu: %s (seed %lld, %zu refs)", jobs.size() + 1, job->cli.output_path.c_str(),
                     (long long)job->gen.seed, job->gen.ref_image_paths.size());
            jobs.push_back(std::move(job));
            continue;
        }
        if (!load_generation_inputs(job->cli.mode, job->gen)) {
            LOG_ERROR("--multi-jobs line %d: loading the inputs failed", line_no);
            return 1;
        }
        LOG_INFO("[multi] queued #%zu: %s (%d frames, seed %lld)", jobs.size() + 1, job->cli.output_path.c_str(),
                 job->gen.video_frames, (long long)job->gen.seed);
        jobs.push_back(std::move(job));
    }
    if (jobs.empty()) {
        LOG_ERROR("--multi-jobs file '%s' has no jobs", base_cli.multi_jobs_path.c_str());
        return 1;
    }
    if (base_cli.mode == IMG_GEN) {
        return run_multi_image_jobs(jobs, base_cli, base_ctx, cache_size_given);
    }

    sd_ctx_params_t sd_ctx_params = base_ctx.to_sd_ctx_params_t(base_cli.taesd_preview);
    SDCtxPtr sd_ctx(new_sd_ctx(&sd_ctx_params));
    if (sd_ctx == nullptr) {
        LOG_INFO("new_sd_ctx_t failed");
        return 1;
    }
    std::vector<const sd_vid_gen_params_t*> params;
    for (auto& job : jobs) {
        SDGenerationParams& gen = job->gen;
        if (gen.sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            gen.sample_params.sample_method = sd_get_default_sample_method(sd_ctx.get());
        }
        if (gen.high_noise_sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            gen.high_noise_sample_params.sample_method = sd_get_default_sample_method(sd_ctx.get());
        }
        if (gen.sample_params.scheduler == SCHEDULER_COUNT) {
            gen.sample_params.scheduler = sd_get_default_scheduler(sd_ctx.get(), gen.sample_params.sample_method);
        }
        job->vid = gen.to_sd_vid_gen_params_t();
        params.push_back(&job->vid);
    }
    MultiJobState state;
    state.jobs = &jobs;
    generate_videos(sd_ctx.get(), params.data(), static_cast<int>(params.size()), multi_job_result, &state);
    LOG_INFO("[multi] %d/%zu jobs saved", state.saved, jobs.size());
    return state.saved == static_cast<int>(jobs.size()) ? 0 : 1;
}

int main(int argc, const char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--version") {
        std::cout << version_string() << "\n";
        return EXIT_SUCCESS;
    }

    SDCliParams cli_params;
    SDContextParams ctx_params;
    ctx_params.conditioning_cache_size = 0;
    SDGenerationParams gen_params;

    sd_set_log_callback(sd_log_cb, (void*)&cli_params);
    parse_args(argc, argv, cli_params, ctx_params, gen_params);

    if (cli_params.mode == METADATA) {
        MetadataReadOptions options;
        options.output_format      = cli_params.metadata_format == "json"
                                         ? MetadataOutputFormat::JSON
                                         : MetadataOutputFormat::TEXT;
        options.include_raw        = cli_params.metadata_raw;
        options.brief              = cli_params.metadata_brief;
        options.include_structural = cli_params.metadata_all;

        std::string error;
        if (!print_image_metadata(cli_params.image_path, options, std::cout, error)) {
            LOG_ERROR("%s", error.c_str());
            return 1;
        }
        return 0;
    }

    if (!gen_params.ad_model_path.empty() && cli_params.mode != IMG_GEN && cli_params.mode != ADETAILER) {
        LOG_ERROR("--ad-model is only supported in image generation and adetailer modes");
        return 1;
    }

    if (gen_params.video_frames > 4) {
        size_t last_dot_pos   = cli_params.preview_path.find_last_of(".");
        std::string base_path = cli_params.preview_path;
        std::string file_ext  = "";
        if (last_dot_pos != std::string::npos) {  // filename has extension
            base_path = cli_params.preview_path.substr(0, last_dot_pos);
            file_ext  = cli_params.preview_path.substr(last_dot_pos);
            std::transform(file_ext.begin(), file_ext.end(), file_ext.begin(), ::tolower);
        }
        if (file_ext == ".png") {
            cli_params.preview_path = base_path + ".avi";
        }
    }
    cli_params.preview_fps = gen_params.fps;

    sd_set_preview_callback(step_callback,
                            cli_params.preview_method,
                            cli_params.preview_interval,
                            !cli_params.preview_noisy,
                            cli_params.preview_noisy,
                            (void*)&cli_params);

    LOG_VERBOSE("version: %s", version_string().c_str());
    LOG_VERBOSE("%s", sd_get_system_info());
    LOG_VERBOSE("%s", cli_params.to_string().c_str());
    LOG_VERBOSE("%s", ctx_params.to_string().c_str());
    LOG_VERBOSE("%s", gen_params.to_string().c_str());

    if (!cli_params.imatrix_out.empty()) {
        if (fs::exists(cli_params.imatrix_out) &&
            std::find(cli_params.imatrix_in.begin(), cli_params.imatrix_in.end(), cli_params.imatrix_out) == cli_params.imatrix_in.end()) {
            LOG_WARN("imatrix file '%s' already exists and will be overwritten", cli_params.imatrix_out.c_str());
        }
        enable_imatrix_collection();
    }

    for (const auto& in_file : cli_params.imatrix_in) {
        LOG_INFO("loading imatrix from '%s'", in_file.c_str());
        if (!load_imatrix(in_file.c_str())) {
            LOG_WARN("failed to load imatrix from '%s'", in_file.c_str());
        }
    }

    if (!cli_params.multi_jobs_path.empty()) {
        return run_multi_jobs(argc, argv, cli_params, ctx_params);
    }

    if (cli_params.mode == CONVERT) {
        bool success = convert_with_components(ctx_params.model_path.c_str(),
                                               ctx_params.clip_l_path.c_str(),
                                               ctx_params.clip_g_path.c_str(),
                                               ctx_params.t5xxl_path.c_str(),
                                               ctx_params.diffusion_model_path.c_str(),
                                               ctx_params.vae_path.c_str(),
                                               cli_params.output_path.c_str(),
                                               ctx_params.wtype,
                                               ctx_params.tensor_type_rules.c_str(),
                                               cli_params.convert_name,
                                               ctx_params.n_threads);
        if (!success) {
            LOG_ERROR("convert '%s'/'%s' to '%s' failed",
                      ctx_params.model_path.c_str(),
                      ctx_params.vae_path.c_str(),
                      cli_params.output_path.c_str());
            return 1;
        } else {
            LOG_INFO("convert '%s'/'%s' to '%s' success",
                     ctx_params.model_path.c_str(),
                     ctx_params.vae_path.c_str(),
                     cli_params.output_path.c_str());
            return 0;
        }
    }

    if (!load_generation_inputs(cli_params.mode, gen_params)) {
        return 1;
    }

    sd_ctx_params_t sd_ctx_params = ctx_params.to_sd_ctx_params_t(cli_params.taesd_preview);

    SDImageVec results;
    int num_results             = 0;
    sd_audio_t* generated_audio = nullptr;

    if (cli_params.mode == UPSCALE) {
        num_results = 1;
        results.push_back(gen_params.init_image.release());
    } else {
        SDCtxPtr sd_ctx(new_sd_ctx(&sd_ctx_params));

        if (sd_ctx == nullptr) {
            LOG_INFO("new_sd_ctx_t failed");
            return 1;
        }

        if (gen_params.sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            gen_params.sample_params.sample_method = sd_get_default_sample_method(sd_ctx.get());
        }

        if (gen_params.high_noise_sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            gen_params.high_noise_sample_params.sample_method = sd_get_default_sample_method(sd_ctx.get());
        }

        if (gen_params.sample_params.scheduler == SCHEDULER_COUNT) {
            gen_params.sample_params.scheduler = sd_get_default_scheduler(sd_ctx.get(), gen_params.sample_params.sample_method);
        }

        sd_img_gen_params_t img_gen_params{};
        const bool use_img_gen_params = cli_params.mode == IMG_GEN || cli_params.mode == ADETAILER;
        if (use_img_gen_params) {
            img_gen_params = gen_params.to_sd_img_gen_params_t();
        }

        if (cli_params.mode == IMG_GEN) {
            sd_image_t* generated_images = nullptr;
            if (!generate_image(sd_ctx.get(), &img_gen_params, &generated_images, &num_results)) {
                generated_images = nullptr;
                num_results      = 0;
            }
            results.adopt(generated_images, num_results);
        } else if (cli_params.mode == ADETAILER) {
            num_results = 1;
            results.push_back(gen_params.init_image.release());
        } else if (cli_params.mode == VID_GEN) {
            sd_vid_gen_params_t vid_gen_params = gen_params.to_sd_vid_gen_params_t();
            sd_image_t* generated_video        = nullptr;
            if (!generate_video(sd_ctx.get(), &vid_gen_params, &generated_video, &num_results, &generated_audio, &cli_params.preview_fps)) {
                generated_video = nullptr;
            }
            gen_params.fps = cli_params.preview_fps;
            results.adopt(generated_video, num_results);
        }

        if (!results) {
            LOG_ERROR("generate failed");
            return 1;
        }

        if (use_img_gen_params &&
            !apply_adetailer(sd_ctx.get(),
                             sd_ctx_params,
                             ctx_params,
                             gen_params,
                             img_gen_params,
                             cli_params.mode,
                             results,
                             num_results)) {
            return 1;
        }
    }

    apply_upscaler(ctx_params, gen_params, sd_ctx_params, results, num_results);

    if (!save_results(cli_params, ctx_params, gen_params, results.data(), num_results, generated_audio)) {
        free_sd_audio(generated_audio);
        return 1;
    }

    if (!cli_params.imatrix_out.empty()) {
        LOG_INFO("saving imatrix to '%s'", cli_params.imatrix_out.c_str());
        save_imatrix(cli_params.imatrix_out.c_str());
    }

    free_sd_audio(generated_audio);

    return 0;
}
