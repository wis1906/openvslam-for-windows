#include "openvslam/data/frame.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/camera_database.h"
#include "openvslam/data/bow_database.h"
#include "openvslam/data/map_database.h"
#include "openvslam/io/map_database_io.h"

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

namespace openvslam {
namespace io {

map_database_io::map_database_io(data::camera_database* cam_db, data::map_database* map_db,
                                 data::bow_database* bow_db, data::bow_vocabulary* bow_vocab)
    : cam_db_(cam_db), map_db_(map_db), bow_db_(bow_db), bow_vocab_(bow_vocab) {}

void map_database_io::save_pcd_pack(const std::string& path) {
	std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

	assert(cam_db_ && map_db_);

	spdlog::info("saving landmarks as pcl to {}", path);

	std::ofstream ofs(path, std::ios::out | std::ios::binary);
	if (!ofs.is_open()) {
		spdlog::critical("cannot create a file at {}", path);
		return;
	}

	std::vector<Vec3_t> landmarks;
	map_db_->to_pcd(landmarks);


	ofs << "VERSION .7" << std::endl;
	ofs << "FIELDS x y z" << std::endl;
	ofs << "SIZE 4 4 4" << std::endl;
	ofs << "TYPE F F F" << std::endl;
	ofs << "COUNT 1 1 1" << std::endl;
	ofs << "WIDTH " + std::to_string(landmarks.size()) << std::endl;
	ofs << "HEIGHT 1" << std::endl;
	ofs << "VIEWPOINT 0 0 0 1 0 0 0" << std::endl;
	ofs << "POINTS " + std::to_string(landmarks.size()) << std::endl;
	ofs << "DATA ascii" << std::endl;

	for (int i = 0; i < landmarks.size(); i++)
	{
		float x = (float)landmarks[i].x();
		float y = (float)landmarks[i].y();
		float z = (float)landmarks[i].z();
		ofs << std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z) << std::endl;
	}

	ofs.close();
	spdlog::info("saving landmarks as pcl successfully");
}

void map_database_io::save_ply_pack(const std::string& path) {
	std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

	assert(cam_db_ && map_db_);

	spdlog::info("saving landmarks as ply to {}", path);

	std::ofstream ofs(path, std::ios::out | std::ios::binary);
	if (!ofs.is_open()) {
		spdlog::critical("cannot create a file at {}", path);
		return;
	}

	std::vector<Vec3_t> landmarks;
	map_db_->to_pcd(landmarks);


	ofs << "ply" << std::endl;
	ofs << "format ascii 1.0" << std::endl;
	ofs << "element vertex " + std::to_string(landmarks.size()) << std::endl;
	ofs << "property float x" << std::endl;
	ofs << "property float y" << std::endl;
	ofs << "property float z" << std::endl;
	ofs << "end_header" << std::endl;

	for (int i = 0; i < landmarks.size(); i++)
	{
		float x = (float)landmarks[i].x();
		float y = (float)landmarks[i].y();
		float z = (float)landmarks[i].z();
		ofs << std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z) << std::endl;
	}

	ofs.close();
	spdlog::info("saving landmarks as ply successfully");
}

void map_database_io::save_custom_pack(const std::string& path) {
	std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

	assert(cam_db_ && map_db_);

	spdlog::info("saving landmarks as custom mapdata to {}", path);

	std::ofstream ofs(path, std::ios::out | std::ios::binary);
	if (!ofs.is_open()) {
		spdlog::critical("cannot create a file at {}", path);
		return;
	}

	std::vector<Mat44_t> kfCameras;
	std::vector<int> camKfIds;
	std::vector<int> camOfIds;

	std::vector<Vec3_t> landmarks;
	std::vector<int> ldmkKfIds;
	std::vector<int> ldmkOfIds;
	map_db_->to_custom_mapdata(kfCameras, camKfIds, camOfIds, landmarks, ldmkKfIds, ldmkOfIds);

	// set header
	ofs << std::to_string(kfCameras.size()) << std::endl;
	ofs << std::to_string(landmarks.size()) << std::endl;
	ofs << std::endl;

	// set camera data
	for (int i = 0; i < kfCameras.size(); i++)
	{
		int kid = camKfIds[i];
		int oid = camOfIds[i];
		Mat44_t camPos = kfCameras[i];
		ofs << std::to_string(kid) + " " + std::to_string(oid) << std::endl;
		ofs << std::to_string(camPos(0, 0)) + " " + std::to_string(camPos(0, 1)) + " " + std::to_string(camPos(0, 2)) + " " + std::to_string(camPos(0, 3)) << std::endl;
		ofs << std::to_string(camPos(1, 0)) + " " + std::to_string(camPos(1, 1)) + " " + std::to_string(camPos(1, 2)) + " " + std::to_string(camPos(0, 3)) << std::endl;
		ofs << std::to_string(camPos(2, 0)) + " " + std::to_string(camPos(2, 1)) + " " + std::to_string(camPos(2, 2)) + " " + std::to_string(camPos(0, 3)) << std::endl;

	}
	ofs << std::endl;

	// set landmark data
	for (int i = 0; i < landmarks.size(); i++)
	{
		int kfid = ldmkKfIds[i];
		int ofid = ldmkOfIds[i];
		float x = (float)landmarks[i].x();
		float y = (float)landmarks[i].y();
		float z = (float)landmarks[i].z();
		ofs << std::to_string(kfid) + " " + std::to_string(ofid) + " " + std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z) << std::endl;
	}

	ofs.close();
	spdlog::info("saving landmarks as custom mapdata successfully");
}

void map_database_io::save_message_pack(const std::string& path) {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

    assert(cam_db_ && map_db_);
    const auto cameras = cam_db_->to_json();
    nlohmann::json keyfrms;
    nlohmann::json landmarks;
    map_db_->to_json(keyfrms, landmarks);

    nlohmann::json json{{"cameras", cameras},
                        {"keyframes", keyfrms},
                        {"landmarks", landmarks},
                        {"frame_next_id", static_cast<unsigned int>(data::frame::next_id_)},
                        {"keyframe_next_id", static_cast<unsigned int>(data::keyframe::next_id_)},
                        {"landmark_next_id", static_cast<unsigned int>(data::landmark::next_id_)}};

    std::ofstream ofs(path, std::ios::out | std::ios::binary);

    if (ofs.is_open()) {
        spdlog::info("save the MessagePack file of database to {}", path);
        const auto msgpack = nlohmann::json::to_msgpack(json);
        ofs.write(reinterpret_cast<const char*>(msgpack.data()), msgpack.size() * sizeof(uint8_t));
        ofs.close();
    }
    else {
        spdlog::critical("cannot create a file at {}", path);
    }
}

void map_database_io::load_message_pack(const std::string& path) {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

    // 1. initialize database

    assert(cam_db_ && map_db_ && bow_db_ && bow_vocab_);
    map_db_->clear();
    bow_db_->clear();

    // 2. load binary bytes

    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        spdlog::critical("cannot load the file at {}", path);
        throw std::runtime_error("cannot load the file at " + path);
    }

    spdlog::info("load the MessagePack file of database from {}", path);
    std::vector<uint8_t> msgpack;
    while (true) {
        uint8_t buffer;
        ifs.read(reinterpret_cast<char*>(&buffer), sizeof(uint8_t));
        if (ifs.eof()) {
            break;
        }
        msgpack.push_back(buffer);
    }
    ifs.close();

    // 3. parse into JSON

    const auto json = nlohmann::json::from_msgpack(msgpack);

    // 4. load database

    // load static variables
    data::frame::next_id_ = json.at("frame_next_id").get<unsigned int>();
    data::keyframe::next_id_ = json.at("keyframe_next_id").get<unsigned int>();
    data::landmark::next_id_ = json.at("landmark_next_id").get<unsigned int>();
    // load database
    const auto json_cameras = json.at("cameras");
    cam_db_->from_json(json_cameras);
    const auto json_keyfrms = json.at("keyframes");
    const auto json_landmarks = json.at("landmarks");
    map_db_->from_json(cam_db_, bow_vocab_, bow_db_, json_keyfrms, json_landmarks);
    const auto keyfrms = map_db_->get_all_keyframes();
    for (const auto keyfrm : keyfrms) {
        bow_db_->add_keyframe(keyfrm);
    }
}

} // namespace io
} // namespace openvslam
