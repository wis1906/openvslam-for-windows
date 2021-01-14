#ifdef USE_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#elif USE_SOCKET_PUBLISHER
#include "socket_publisher/publisher.h"
#endif

#include "openvslam/system.h"
#include "openvslam/config.h"

#include <iostream>
#include <chrono>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>

#ifdef USE_STACK_TRACE_LOGGER
#include <glog/logging.h>
#endif

#ifdef USE_GOOGLE_PERFTOOLS
#include <gperftools/profiler.h>
#endif

void mono_tracking(const std::shared_ptr<openvslam::config>& cfg,
	const std::string& vocab_file_path, const std::string& video_file_path, const std::string& mask_img_path,
	const unsigned int frame_skip, const bool no_sleep, const bool auto_term,
	const bool eval_log, const std::string& map_db_path, const std::string& map_pcd_path, const std::string& map_ply_path);

int main(int argc, char* argv[]) {
#ifdef USE_STACK_TRACE_LOGGER
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
#endif
    std::string vocab_file_path = "../inputs/orb_vocab.dbow2";
    std::string video_file_path = "../inputs/aist_living_lab_1/video.mp4 ";
    std::string config_file_path = "../inputs/aist_living_lab_1/config.yaml";
    std::string map_db_path = "../inputs/aist_living_lab_1/map.msg";
    std::string map_pcd_path = "../inputs/aist_living_lab_1/map.pcd";
    std::string map_ply_path = "../inputs/aist_living_lab_1/map.ply";
    std::string mask_img_path = "";
    int frame_skip = 1; // default=1
    bool no_sleep = false;
    bool auto_term = false;
    bool debug_mode = false;
    bool eval_log = true;

    // check validness of options
    if (vocab_file_path == "" || video_file_path == "" || config_file_path == "") {
        std::cerr << "invalid arguments" << std::endl;
        return EXIT_FAILURE;
    }

    // setup logger
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
    if (debug_mode) {
        spdlog::set_level(spdlog::level::debug);
    }
    else {
        spdlog::set_level(spdlog::level::info);
    }

    // load configuration
    std::shared_ptr<openvslam::config> cfg;
    try {
        cfg = std::make_shared<openvslam::config>(config_file_path);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStart("slam.prof");
#endif
    // run tracking
    if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Monocular) {
        mono_tracking(cfg, vocab_file_path, video_file_path, mask_img_path,
                      frame_skip, no_sleep, auto_term,
                      eval_log, map_db_path, map_pcd_path, map_ply_path);
    }
    else {
        throw std::runtime_error("Invalid setup type: " + cfg->camera_->get_setup_type_string());
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStop();
#endif
    return EXIT_SUCCESS;
}


void mono_tracking(const std::shared_ptr<openvslam::config>& cfg,
	const std::string& vocab_file_path, const std::string& video_file_path, const std::string& mask_img_path,
	const unsigned int frame_skip, const bool no_sleep, const bool auto_term,
	const bool eval_log, const std::string& map_db_path, const std::string& map_pcd_path, const std::string& map_ply_path) {
	// load the mask image
	const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);

	// build a SLAM system
	openvslam::system SLAM(cfg, vocab_file_path);
	// startup the SLAM process
	SLAM.startup();

	// create a viewer object
	// and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
	pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
	socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

	auto video = cv::VideoCapture(video_file_path, cv::CAP_FFMPEG);
	std::vector<double> track_times;

	cv::Mat frame;
	double timestamp = 0.0;

	unsigned int num_frame = 0;

	bool is_not_end = true;
	// run the SLAM in another thread
	std::thread thread([&]() {
		while (is_not_end) {
			is_not_end = video.read(frame);

			const auto tp_1 = std::chrono::steady_clock::now();

			if (!frame.empty() && (num_frame % frame_skip == 0)) {
				// input the current frame and estimate the camera pose
				SLAM.feed_monocular_frame(frame, timestamp, mask);
			}

			const auto tp_2 = std::chrono::steady_clock::now();

			const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
			if (num_frame % frame_skip == 0) {
				track_times.push_back(track_time);
			}

			// wait until the timestamp of the next frame
			if (!no_sleep) {
				const auto wait_time = 1.0 / cfg->camera_->fps_ - track_time;
				if (0.0 < wait_time) {
					std::this_thread::sleep_for(std::chrono::microseconds(static_cast<unsigned int>(wait_time * 1e6)));
				}
			}

			timestamp += 1.0 / cfg->camera_->fps_;
			++num_frame;

			// check if the termination of SLAM system is requested or not
			if (SLAM.terminate_is_requested()) {
				break;
			}
		}

		// wait until the loop BA is finished
		while (SLAM.loop_BA_is_running()) {
			std::this_thread::sleep_for(std::chrono::microseconds(5000));
		}

		// automatically close the viewer
#ifdef USE_PANGOLIN_VIEWER
		if (auto_term) {
			viewer.request_terminate();
		}
#elif USE_SOCKET_PUBLISHER
		if (auto_term) {
			publisher.request_terminate();
		}
#endif
	});

	// run the viewer in the current thread
#ifdef USE_PANGOLIN_VIEWER
	viewer.run();
#elif USE_SOCKET_PUBLISHER
	publisher.run();
#endif

	thread.join();

	// shutdown the SLAM process
	SLAM.shutdown();

	if (eval_log) {
		// output the trajectories for evaluation
		SLAM.save_frame_trajectory("frame_trajectory.txt", "TUM");
		SLAM.save_keyframe_trajectory("keyframe_trajectory.txt", "TUM");
		// output the tracking times for evaluation
		std::ofstream ofs("track_times.txt", std::ios::out);
		if (ofs.is_open()) {
			for (const auto track_time : track_times) {
				ofs << track_time << std::endl;
			}
			ofs.close();
		}
	}

	// output the map database
	if (!map_db_path.empty())
		SLAM.save_map_database(map_db_path);
	if (!map_pcd_path.empty())
		SLAM.save_pcd_database(map_pcd_path);
	if (!map_ply_path.empty())
		SLAM.save_ply_database(map_ply_path);

	std::sort(track_times.begin(), track_times.end());
	const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
	std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
	std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
}