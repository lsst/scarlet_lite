#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <math.h>
#include <algorithm>
#include <stack>
#include <queue>
#include <vector>
#include <utility> // For std::pair
#include <stdexcept>
#include <iostream>

namespace py = pybind11;
using namespace pybind11::literals;

typedef Eigen::Array<int, 4, 1> Bounds;
typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixB;


// Create a boolean mask `footprint` for all of the pixels that are connected to the pixel
// located at `i,j` and create the bounding box for the `footprint` in `image`.
template <typename M>
void get_connected_pixels(
    const int start_i,
    const int start_j,
    py::EigenDRef<const M> image,
    py::EigenDRef<MatrixB> unchecked,
    py::EigenDRef<MatrixB> footprint,
    Eigen::Ref<Bounds> bounds,
    const double thresh=0
){
    std::stack<std::pair<int, int>> stack;
    stack.push(std::make_pair(start_i, start_j));

    while (!stack.empty()) {
        int i, j;
        std::tie(i, j) = stack.top();
        stack.pop();

        if (!unchecked(i, j)) {
            continue;
        }
        unchecked(i, j) = false;

        if (image(i, j) > thresh) {
            footprint(i, j) = true;

            if (i < bounds[0]) {
                bounds[0] = i;
            } else if (i > bounds[1]) {
                bounds[1] = i;
            }
            if (j < bounds[2]) {
                bounds[2] = j;
            } else if (j > bounds[3]) {
                bounds[3] = j;
            }

            if (i > 0 && unchecked(i-1, j)) {
                stack.push(std::make_pair(i-1, j));
            }
            if (i < image.rows() - 1 && unchecked(i+1, j)) {
                stack.push(std::make_pair(i+1, j));
            }
            if (j > 0 && unchecked(i, j-1)) {
                stack.push(std::make_pair(i, j-1));
            }
            if (j < image.cols() - 1 && unchecked(i, j+1)) {
                stack.push(std::make_pair(i, j+1));
            }
        }
    }
}


/// Proximal operator to trim pixels not connected to one of the source centers.
template <typename M>
MatrixB get_connected_multipeak(
    py::EigenDRef<const M> image,
    const std::vector<std::vector<int>>& centers,
    const double thresh=0
){
    const int height = image.rows();
    const int width = image.cols();
    MatrixB footprint = MatrixB::Zero(height, width);
    std::queue<std::pair<int, int>> pixel_queue;

    // Seed the queue with peaks
    for(const auto& center : centers){
        const int y = center[0];
        const int x = center[1];

        // Validate center coordinates
        if (y < 0 || y >= height || x < 0 || x >= width) {
            throw std::out_of_range("Center coordinates (" + std::to_string(y) + ", " +
                                  std::to_string(x) + ") are out of image bounds [0, " +
                                  std::to_string(height) + ") x [0, " + std::to_string(width) + ")");
        }

        if (!footprint(y, x) && image(y, x) > thresh) {
            footprint(y, x) = true;
            pixel_queue.emplace(y, x);
        }
    }

    // 4-connectivity offsets
    const std::vector<std::pair<int, int>> offsets = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    // Flood fill
    while (!pixel_queue.empty()) {
        auto [i, j] = pixel_queue.front();
        pixel_queue.pop();

        for (const auto& [di, dj] : offsets) {
            int ni = i + di;
            int nj = j + dj;
            if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                if (!footprint(ni, nj) && image(ni, nj) > thresh) {
                    footprint(ni, nj) = true;
                    pixel_queue.emplace(ni, nj);
                }
            }
        }
    }

    return footprint;
}


/// A Peak in a Footprint
/// This class is meant to keep track of both the position and
/// flux at the location of a maximum in a Footprint
class Peak {
public:
    Peak(int y, int x, double flux){
        _y = y;
        _x = x;
        _flux = flux;
    }

    int getY(){
        return _y;
    }

    int getX(){
        return _x;
    }

    double getFlux(){
        return _flux;
    }


private:
    int _y;
    int _x;
    double _flux;
};


/// Sort two peaks, placing the brightest peak first
bool sortBrightness(Peak a, Peak b){
    return a.getFlux() > b.getFlux();
}


// Get a list of peaks found in an image.
// To make ut easier to cull peaks that are too close together
// and ensure that every footprint has at least one peak,
// this algorithm is meant to be run on a single footprint
// created by `get_connected_pixels`.
template <typename M>
std::vector<Peak> get_peaks(
    M& image,
    const double min_separation,
    const double peak_thresh,
    const int y0,
    const int x0
){
    const int height = image.rows();
    const int width = image.cols();

    std::vector<Peak> peaks;

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            if(image(i, j) < peak_thresh){
                continue;
            }
            if(i > 0 && image(i, j) <= image(i-1, j)){
                continue;
            }
            if(i < height-1 && image(i,j) <= image(i+1, j)){
                continue;
            }
            if(j > 0 && image(i, j) <= image(i, j-1)){
                continue;
            }
            if(j < width-1 && image(i,j) <= image(i, j+1)){
                continue;
            }

            if(i > 0 && j > 0 && image(i, j) <= image(i-1, j-1)){
                continue;
            }
            if(i < height-1 && j < width-1 && image(i,j) <= image(i+1, j+1)){
                continue;
            }
            if(i < height-1 && j > 0 && image(i, j) <= image(i+1, j-1)){
                continue;
            }
            if(i > 0 && j < width-1 && image(i,j) <= image(i-1, j+1)){
                continue;
            }

            peaks.push_back(Peak(i+y0, j+x0, static_cast<double>(image(i, j))));
        }
    }

    if(peaks.empty()){
        return peaks;
    }

    /// Sort the peaks in the footprint so that the brightest are first
    std::sort (peaks.begin(), peaks.end(), sortBrightness);

    // Remove peaks within min_separation
    double min_separation2 = min_separation * min_separation;
    for (size_t i = 0; i < peaks.size() - 1; ++i) {
        for (size_t j = i + 1; j < peaks.size();) {
            Peak *p1 = &peaks[i];
            Peak *p2 = &peaks[j];
            double dy = p1->getY() - p2->getY();
            double dx = p1->getX() - p2->getX();
            double separation2 = dy*dy + dx*dx;
            if (separation2 < min_separation2) {
                peaks.erase(peaks.begin() + j);
            } else {
                ++j;
            }
        }
    }
    return peaks;
}


// A detected footprint
class Footprint {
public:
    Footprint(MatrixB footprint, std::vector<Peak> peaks, Bounds bounds){
        _data = footprint;
        this->peaks = peaks;
        _bounds = bounds;
    }

    MatrixB getFootprint(){
        return _data;
    }

    std::vector<Peak> peaks;

    Bounds getBounds(){
        return _bounds;
    }

    void addPeak(Peak peak){
        peaks.push_back(peak);
    }

private:
    MatrixB _data;
    Bounds _bounds;
};


template <typename M>
void maskImage(
    py::EigenDRef<M> image,
    py::EigenDRef<MatrixB> footprint
){
    const int height = image.rows();
    const int width = image.cols();

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            if(!footprint(i,j)){
                image(i,j) = 0;
            }
        }
    }
}

/**
 * Get all footprints in an image
 *
 * @param image: The image to search for footprints
 * @param min_separation: The minimum separation (in pixels) between peaks in a footprint
 * @param min_area: The minimum area of a footprint in pixels
 * @param peak_thresh: The minimum flux of a peak to be detected.
 * @param footprint_thresh: The minimum flux of a pixel to be included in a footprint
 * @param find_peaks: If True, find peaks in each footprint
 * @param y0: The y-coordinate of the top-left corner of the image
 * @param x0: The x-coordinate of the top-left corner of the image
 *
 * @return: A list of Footprints
 */
template <typename M, typename P>
std::vector<Footprint> get_footprints(
    py::EigenDRef<const M> image,
    const double min_separation,
    const int min_area,
    const double peak_thresh,
    const double footprint_thresh,
    const bool find_peaks=true,
    const int y0=0,
    const int x0=0
){
    const int height = image.rows();
    const int width = image.cols();

    std::vector<Footprint> footprints;
    MatrixB unchecked = MatrixB::Ones(height, width);
    MatrixB footprint = MatrixB::Zero(height, width);

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            Bounds bounds; bounds << i, i, j, j;
            get_connected_pixels(i, j, image, unchecked, footprint, bounds, footprint_thresh);
            int subHeight = bounds[1]-bounds[0]+1;
            int subWidth = bounds[3]-bounds[2]+1;
            if(subHeight * subWidth > min_area){
                MatrixB subFootprint = footprint.block(bounds[0], bounds[2], subHeight, subWidth);
                int area = subFootprint.count();
                if(area >= min_area){
                    M patch = image.block(bounds[0], bounds[2], subHeight, subWidth);
                    maskImage<M>(patch, subFootprint);
                    std::vector<Peak> _peaks;
                    if(find_peaks){
                        _peaks = get_peaks(
                            patch,
                            min_separation,
                            peak_thresh,
                            bounds[0] + y0,
                            bounds[2] + x0
                        );
                    }
                    // Only add footprints that have at least one peak above the
                    // minimum peak_thresh.
                    if(!_peaks.empty() || !find_peaks){
                        Bounds trueBounds; trueBounds << bounds[0] + y0,
                            bounds[1] + y0, bounds[2] + x0, bounds[3] + x0;
                        footprints.push_back(Footprint(subFootprint, _peaks, trueBounds));
                    }
                }
            }
            footprint.block(bounds[0], bounds[2], subHeight, subWidth) = MatrixB::Zero(subHeight, subWidth);
        }
    }
    return footprints;
}



PYBIND11_MODULE(detect_pybind11, mod) {
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixF;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixD;

  mod.doc() = "Fast detection algorithms implemented in C++";

  mod.def("get_connected_pixels", &get_connected_pixels<MatrixB>,
          "Create a boolean mask for pixels that are connected");
  mod.def("get_connected_pixels", &get_connected_pixels<MatrixF>,
          "Create a boolean mask for pixels that are connected");
  mod.def("get_connected_pixels", &get_connected_pixels<MatrixD>,
          "Create a boolean mask for pixels that are connected");

  mod.def("get_connected_multipeak", &get_connected_multipeak<MatrixB>,
          "Trim pixels not conencted to a center from a list of centers");
  mod.def("get_connected_multipeak", &get_connected_multipeak<MatrixF>,
          "Trim pixels not conencted to a center from a list of centers");
  mod.def("get_connected_multipeak", &get_connected_multipeak<MatrixD>,
          "Trim pixels not conencted to a center from a list of centers");

  mod.def("get_peaks", &get_peaks<MatrixF>,
          "Get a list of peaks in a footprint created by get_connected_pixels");
  mod.def("get_peaks", &get_peaks<MatrixD>,
          "Get a list of peaks in a footprint created by get_connected_pixels");

  mod.def("get_footprints", &get_footprints<MatrixF, float>,
          "Create a list of all of the footprints in an image, with their peaks",
          "image"_a, "min_separation"_a, "min_area"_a, "peak_thresh"_a, "footprint_thresh"_a,
          "find_peaks"_a=true, "y0"_a=0, "x0"_a=0);
  mod.def("get_footprints", &get_footprints<MatrixD, double>,
          "Create a list of all of the footprints in an image, with their peaks",
          "image"_a, "min_separation"_a, "min_area"_a, "peak_thresh"_a, "footprint_thresh"_a,
          "find_peaks"_a=true, "y0"_a=0, "x0"_a=0);

  py::class_<Footprint>(mod, "Footprint")
        .def(py::init<MatrixB, std::vector<Peak>, Bounds>(),
             "footprint"_a, "peaks"_a, "bounds"_a)
        .def_property_readonly("data", &Footprint::getFootprint)
        .def_readwrite("peaks", &Footprint::peaks)
        .def_property_readonly("bounds", &Footprint::getBounds)
        .def("add_peak", &Footprint::addPeak);

  py::class_<Peak>(mod, "Peak")
        .def(py::init<int, int, double>(),
            "y"_a, "x"_a, "flux"_a)
        .def_property_readonly("y", &Peak::getY)
        .def_property_readonly("x", &Peak::getX)
        .def_property_readonly("flux", &Peak::getFlux);
}
