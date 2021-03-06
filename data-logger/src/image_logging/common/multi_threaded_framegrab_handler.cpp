/*
 * multi_threaded_framegrab_handler.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger/image_logging/common/multi_threaded_framegrab_handler.h"
#include "f1_datalogger/proto/TimestampedImage.pb.h"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <google/protobuf/util/json_util.h>
namespace fs = boost::filesystem;
namespace deepf1
{

MultiThreadedFrameGrabHandler::MultiThreadedFrameGrabHandler(std::string images_folder, unsigned int thread_count) 
: running_(false), counter_(1), images_folder_(images_folder)
{
  fs::path imf(images_folder_);
  if(!fs::is_directory(imf))
  {
    fs::create_directories(imf);
  }
  thread_count_= thread_count;
}

MultiThreadedFrameGrabHandler::~MultiThreadedFrameGrabHandler()
{
  running_ = false;
}

inline bool MultiThreadedFrameGrabHandler::isReady()
{
  return true;
}

void MultiThreadedFrameGrabHandler::handleData(const TimestampedImageData& data)
{
  std::lock_guard<std::mutex> lk(queue_mutex_);
  queue_->push(data);
}
void MultiThreadedFrameGrabHandler::init(const std::chrono::high_resolution_clock::time_point& begin,
                                         const cv::Size& window_size)
{
  begin_ = std::chrono::high_resolution_clock::time_point(begin);
  running_ = true;
  queue_.reset(new tbb::concurrent_queue<TimestampedImageData>);
  thread_pool_.reset(new tbb::task_group);
  for(unsigned int i = 0; i < thread_count_; i ++)
  {
    thread_pool_->run(std::bind(&MultiThreadedFrameGrabHandler::workerFunc_,this));
  }
}

void MultiThreadedFrameGrabHandler::workerFunc_()
{
  std::cout<<"Spawned a worker thread to log images" <<std::endl;
  while( running_ || !queue_->empty() )
  {
    TimestampedImageData data;
    {
      std::lock_guard<std::mutex> lk(queue_mutex_);
      if(queue_->empty())
      {
        continue;
      }
      if(!queue_->try_pop(data))
      {
        continue;
      }
    }
    unsigned long counter = counter_.fetch_and_increment();
    fs::path  images_folder(images_folder_);
    google::protobuf::uint64 delta = (google::protobuf::uint64)(std::chrono::duration_cast<std::chrono::microseconds>(data.timestamp - begin_).count());
	  //std::cout << "Got some image data. Clock Delta = " << delta << std::endl;
	std::string image_file("image_" + std::to_string(counter) + ".jpg");
    cv::imwrite( ( images_folder / fs::path(image_file) ).string() , data.image);


    deepf1::protobuf::TimestampedImage tag;
    tag.set_image_file(image_file);
    tag.set_timestamp(delta);

    std::string pb_filename("image_" + std::to_string(counter) + ".pb");
    std::string pb_output_file = (images_folder / fs::path(pb_filename)).string();
    std::ofstream ostream;
    ostream.open(pb_output_file.c_str(), std::ofstream::out);
    tag.SerializeToOstream(&ostream);
    ostream.flush();
    ostream.close();

    std::string json;
    google::protobuf::util::JsonOptions opshinz;
    opshinz.always_print_primitive_fields = true;
    opshinz.add_whitespace = true;
    google::protobuf::util::MessageToJsonString(tag, &json, opshinz);
    std::string json_file = pb_filename + ".json";
    std::string json_fn = ( images_folder / fs::path(json_file) ).string();
    ostream.open(json_fn.c_str(), std::ofstream::out);
    ostream << json;
    ostream.flush();
    ostream.close();
  }
}
const std::string MultiThreadedFrameGrabHandler::getImagesFolder() const
{
  return images_folder_;
}
} /* namespace deepf1 */
