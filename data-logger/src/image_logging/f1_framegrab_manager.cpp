/*
 * F1FrameGrabManager.cpp
 *
 *  Created on: Dec 4, 2018
 *      Author: ttw2xk
 */

#include "image_logging/f1_framegrab_manager.h"
#include "image_logging/utils/opencv_utils.h"
namespace scl = SL::Screen_Capture;
namespace deepf1
{
F1FrameGrabManager::F1FrameGrabManager(std::shared_ptr<scl::Window> window,
                                       std::shared_ptr<std::chrono::high_resolution_clock> clock,
                                       std::shared_ptr<IF1FrameGrabHandler> capture_handler)
{
  capture_handler_ = capture_handler;
  clock_ = clock;
  window_ = window;
  capture_config_ = scl::CreateCaptureConfiguration( (scl::WindowCallback)std::bind(&F1FrameGrabManager::get_windows_, this));
  capture_config_->onNewFrame((scl::WindowCaptureCallback)std::bind(&F1FrameGrabManager::onNewFrame_, this, std::placeholders::_1, std::placeholders::_2));
}
F1FrameGrabManager::~F1FrameGrabManager()
{
}
std::vector<scl::Window> F1FrameGrabManager::get_windows_()
{
  return std::vector<scl::Window> {*window_};
}
void F1FrameGrabManager::onNewFrame_(const scl::Image &img, const scl::Window &monitor)
{
  if(capture_handler_->isReady())
  {
    TimestampedImageData timestamped_image;
    timestamped_image.image = deepf1::OpenCVUtils::toCV(img);
    timestamped_image.timestamp = clock_->now();
    capture_handler_->handleData(timestamped_image);
  }
}
void F1FrameGrabManager::start()
{
  capture_manager_ = capture_config_->start_capturing();
  capture_manager_->setFrameChangeInterval(std::chrono::milliseconds(64));
}
} /* namespace deepf1 */
