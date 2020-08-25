#include <iostream>
#include <iomanip> 
#include <string>
#include <vector>
#include <fstream>
#include <thread>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <opencv2/core/eigen.hpp>
#define OPENCV

#include "yolo_v2_class.hpp"    // imported functions from DLL

#ifdef OPENCV
#include <opencv2/opencv.hpp>           // C++
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio.hpp"
#pragma comment(lib, "opencv_world320.lib")  
#else
#pragma comment(lib, "opencv_core2413.lib")  
#pragma comment(lib, "opencv_imgproc2413.lib")  
#pragma comment(lib, "opencv_highgui2413.lib") 
#endif

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names, unsigned int wait_msec = 0) {
    for (auto &i : result_vec) {
        cv::Scalar color(60, 160, 260);
        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 3);
        if(obj_names.size() > i.obj_id)
            putText(mat_img, obj_names[i.obj_id], cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
        if(i.track_id > 0)
            putText(mat_img, std::to_string(i.track_id), cv::Point2f(i.x+5, i.y + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
    }
    cv::imshow("window name", mat_img);
    cv::waitKey(wait_msec);
}
#endif  // OPENCV


void show_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names) {
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
        std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y 
            << ", w = " << i.w << ", h = " << i.h
            << std::setprecision(3) << ", prob = " << i.prob << std::endl;
    }
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; file >> line;) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}

void read_parm(std::string cparm, Eigen::Matrix3d &cmtx, Eigen:: Matrix4d &tmtx){
	cv::Mat cmat,tmat;
	cv::FileStorage fs(cparm, cv::FileStorage::READ);
	fs["camera_matrix"] >> cmat;
    fs["trans_matrix"] >> tmat;
	cv::cv2eigen(cmat, cmtx);
	cv::cv2eigen(tmat, tmtx);
}

void point_filters(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud){
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);          
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

	pcl::SACSegmentation<pcl::PointXYZI> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(0.5);
	seg.setInputCloud(cloud);
	seg.segment(*inliers,*coefficients);

	pcl::ExtractIndices<pcl::PointXYZI> extra;
	extra.setInputCloud(cloud);
	extra.setIndices(inliers);
	extra.setNegative(true);
	extra.filter(*cloud);

	pcl::RadiusOutlierRemoval<pcl::PointXYZI> outlier;
	outlier.setInputCloud(cloud);
	outlier.setRadiusSearch(0.3);
	outlier.setMinNeighborsInRadius(6);
	outlier.filter(*cloud);

	outlier.setInputCloud(cloud);
	outlier.setRadiusSearch(0.3);
	outlier.setMinNeighborsInRadius(3);
	outlier.filter(*cloud);

	pcl::VoxelGrid<pcl::PointXYZI> voxel;
	voxel.setLeafSize(0.15f, 0.15f, 0.15f);
	voxel.setInputCloud(cloud);
	voxel.filter(*cloud);
}	

void cluster_point(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, Eigen::Matrix3d cmtx, int x, int y, int w, int h, int count){
	pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
	for(auto &pt : *cloud){
		Eigen::MatrixXd pn(3,1);
		Eigen::MatrixXd rs(3,1);
		int u,v; 
		pn(0,0) = pt.x/pt.z;
		pn(1,0) = pt.y/pt.z;
		pn(2,0) = 1.0;
		rs = cmtx*pn;
		if(x <rs(0,0) && rs(0,0) <x+w && y < rs(1,0) && rs(1,0) < y+h && pt.z < 20){
			cluster->push_back(pt);
		}
		} 
	pcl::PCDWriter witer;
	if(!cluster->empty() && cluster->size() > 10){
		witer.write("test_"+std::to_string(count)+".pcd", *cluster);
		std::cout << "count" << count <<std::endl;
	} 
}

int main(int args, char* argv[]){
	Detector detector("/home/htk/src/yolo_datas/kittiv4-tiny.cfg", "/home/htk/src/yolo_datas/weights/kittiv4-tiny_final.weights");
    //Detector detector();

    auto obj_names = objects_names_from_file("/home/htk/src/yolo_datas/datas_v2/classes.txt");

	Eigen::Matrix3d cmtx;
	Eigen::Matrix4d tmtx;
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PCDReader reader;
	pcl::PCDWriter writer;
	if(args < 4){
		std::cout << "need 3 argument (ply, img, cparam)" << std::endl;
		return 0;
	} 
	std::string cloud_name = argv[1];	
	std::string img_name = argv[2];	
	std::string cparam_name = argv[3];	

	cv::Mat mat_img = cv::imread(img_name);
    std::vector<bbox_t> result_vec = detector.detect(mat_img);
    draw_boxes(mat_img, result_vec, obj_names);
    show_result(result_vec, obj_names);

	reader.read(cloud_name,*cloud);
	read_parm(cparam_name, cmtx, tmtx);
	Eigen::Matrix4d trans;
	trans << 
	0., -1., 0., 0,
	0., 0,  -1, 0,
	1, 0., 0, 0,
	0, 0, 0, 1;
	pcl::transformPointCloud(*cloud, *cloud, trans);
	pcl::transformPointCloud(*cloud, *cloud, tmtx);
	point_filters(cloud);	
	writer.write("out.pcd",*cloud);
	int count = 0;
	for (auto &i : result_vec){
		cluster_point(cloud,cmtx,i.x, i.y, i.w, i.h,count);
		count ++;
	}
	
	cluster_point(cloud, cmtx, 252, 276, 200, 110,10);
	return 0;
}
