#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <math.h>
#include <algorithm>
#include <limits>
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
/// This class keeps track of the position and flux at the location of a
/// local maximum in a Footprint, together with the `saddle`: the highest
/// level at which the peak's basin connects to the basin of a brighter peak
/// (NaN for the brightest peak in a footprint, which never joins a brighter
/// basin). `flux - saddle` is the peak's prominence, the quantity that
/// `kappa` thresholds in `get_peaks`.
class Peak {
public:
    Peak(int y, int x, double flux, double saddle=std::numeric_limits<double>::quiet_NaN()){
        _y = y;
        _x = x;
        _flux = flux;
        _saddle = saddle;
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

    double getSaddle(){
        return _saddle;
    }


private:
    int _y;
    int _x;
    double _flux;
    double _saddle;
};


/// Sort two peaks, placing the brightest peak first
bool sortBrightness(Peak a, Peak b){
    return a.getFlux() > b.getFlux();
}


/// Find the root of a union-find tree with path compression.
inline int union_find(std::vector<int>& parent, int i){
    int root = i;
    while (parent[root] != root) {
        root = parent[root];
    }
    while (parent[i] != root) {
        int next = parent[i];
        parent[i] = root;
        i = next;
    }
    return root;
}


/**
 * Find the peaks in a masked image with a contrast-limited watershed.
 *
 * Pixels inside `mask` are visited in decreasing order of value. A pixel
 * with no visited 8-neighbor is a strict local maximum and seeds a new basin
 * (and a peak, if it is at least `peak_thresh`). Otherwise it joins the
 * basins of its visited neighbors. When a pixel connects two basins for the
 * first time, its value is the saddle level between them: the basin with the
 * lower maximum is absorbed into the other, and its peak is culled if its
 * prominence above that saddle, `flux - saddle`, is less than `kappa`. Because
 * pixels are processed in decreasing order every later saddle is lower, so a
 * peak that survives its first test survives all of them and each peak is
 * tested exactly once.
 *
 * With `kappa <= 0` nothing is culled and the result is the set of strict
 * 8-connected local maxima above `peak_thresh`.
 *
 * The cost is O(N log N) in the number of masked pixels, independent of the
 * number of peaks.
 */
template <typename M>
std::vector<Peak> watershed_peaks(
    const M& image,
    const MatrixB& mask,
    const double min_separation,
    const double peak_thresh,
    const double kappa,
    const int y0,
    const int x0
){
    const int height = image.rows();
    const int width = image.cols();
    const int n = height * width;

    // Collect the masked pixels and sort them by decreasing value. Ties are
    // broken by pixel index so the result is deterministic; the first pixel of
    // a plateau becomes the seed and the rest join it (no plateau pixel other
    // than the first can be a peak).
    std::vector<std::pair<double, int>> pixels;
    pixels.reserve(mask.count());
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (mask(i, j)) {
                pixels.emplace_back(static_cast<double>(image(i, j)), i * width + j);
            }
        }
    }
    std::vector<Peak> peaks;
    if (pixels.empty()) {
        return peaks;
    }
    std::sort(pixels.begin(), pixels.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });

    // Union-find over pixels. parent == -1 marks an unvisited pixel.
    std::vector<int> parent(n, -1);
    // For each root, the pixel index of the basin's maximum.
    std::vector<int> comp_max(n, -1);
    // For each pixel, the index into `candidates` of the peak it seeded, or -1.
    std::vector<int> peak_id(n, -1);

    struct Candidate {
        int pixel;
        double flux;
        double saddle;
        bool culled;
    };
    std::vector<Candidate> candidates;

    auto value = [&](int idx) { return static_cast<double>(image(idx / width, idx % width)); };

    // Merge the basins rooted at idx_a and idx_b at saddle level pixel_value; returns the new root.
    auto merge = [&](int idx_a, int idx_b, double pixel_value) {
        const double value_a = value(comp_max[idx_a]);
        const double value_b = value(comp_max[idx_b]);
        int hi = idx_a, lo = idx_b;
        if (value_b > value_a) {
            hi = idx_b;
            lo = idx_a;
        }
        const int pk = peak_id[comp_max[lo]];
        if (pk >= 0) {
            Candidate& c = candidates[pk];
            c.saddle = pixel_value;
            if (c.flux - pixel_value < kappa) {
                c.culled = true;
            }
        }
        parent[lo] = hi;
        return hi;
    };

    const int di[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dj[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

    for (const auto& pv : pixels) {
        const double pixel_value = pv.first;
        const int pixel_index = pv.second;
        const int i = pixel_index / width;
        const int j = pixel_index % width;

        int root = -1;
        for (int k = 0; k < 8; ++k) {
            const int ni = i + di[k];
            const int nj = j + dj[k];
            if (ni < 0 || ni >= height || nj < 0 || nj >= width) {
                continue;
            }
            const int neighbor_index = ni * width + nj;
            if (parent[neighbor_index] < 0) {
                continue;  // unvisited, i.e. lower than pixel_index (or masked out)
            }
            const int root_index = union_find(parent, neighbor_index);
            if (root < 0) {
                // First visited neighbor: pixel_index joins its basin.
                parent[pixel_index] = root_index;
                root = root_index;
            } else if (root_index != root) {
                root = merge(root, root_index, pixel_value);
            }
        }
        if (root < 0) {
            // Strict local maximum: seed a new basin.
            parent[pixel_index] = pixel_index;
            comp_max[pixel_index] = pixel_index;
            if (pixel_value >= peak_thresh) {
                peak_id[pixel_index] = static_cast<int>(candidates.size());
                candidates.push_back({pixel_index, pixel_value, std::numeric_limits<double>::quiet_NaN(), false});
            }
        }
    }

    for (const Candidate& c : candidates) {
        if (!c.culled) {
            peaks.push_back(Peak(c.pixel / width + y0, c.pixel % width + x0, c.flux, c.saddle));
        }
    }
    if (peaks.empty()) {
        return peaks;
    }

    /// Sort the peaks in the footprint so that the brightest are first
    std::sort(peaks.begin(), peaks.end(), sortBrightness);

    // Remove peaks within min_separation of a brighter peak. This is a hard
    // floor on top of the contrast test and is skipped when min_separation <= 0.
    if (min_separation > 0) {
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
    }
    return peaks;
}


// Get a list of peaks found in an image.
// This is meant to be run on a single footprint created by
// `get_connected_pixels`. Only pixels above `footprint_thresh` take part in
// the watershed; with the default (-inf) every pixel does.
template <typename M>
std::vector<Peak> get_peaks(
    const M& image,
    const double min_separation,
    const double peak_thresh,
    const int y0,
    const int x0,
    const double kappa=0.0,
    const double footprint_thresh=-std::numeric_limits<double>::infinity()
){
    const int height = image.rows();
    const int width = image.cols();
    MatrixB mask(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            mask(i, j) = static_cast<double>(image(i, j)) > footprint_thresh;
        }
    }
    return watershed_peaks(image, mask, min_separation, peak_thresh, kappa, y0, x0);
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
 * @param kappa: The minimum prominence of a peak above the saddle to a
 *   brighter peak in the same footprint. Peaks with smaller prominence are
 *   culled; 0 keeps every local maximum.
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
    const int x0=0,
    const double kappa=0.0
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
            if(subHeight * subWidth >= min_area){
                MatrixB subFootprint = footprint.block(bounds[0], bounds[2], subHeight, subWidth);
                int area = subFootprint.count();
                if(area >= min_area){
                    std::vector<Peak> _peaks;
                    if(find_peaks){
                        // The watershed only visits masked pixels, so the
                        // patch does not need to be zeroed outside the footprint.
                        M patch = image.block(bounds[0], bounds[2], subHeight, subWidth);
                        _peaks = watershed_peaks(
                            patch,
                            subFootprint,
                            min_separation,
                            peak_thresh,
                            kappa,
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
          "Get a list of peaks in a footprint with a contrast-limited watershed",
          "image"_a, "min_separation"_a, "peak_thresh"_a, "y0"_a, "x0"_a,
          "kappa"_a=0.0, "footprint_thresh"_a=-std::numeric_limits<double>::infinity());
  mod.def("get_peaks", &get_peaks<MatrixD>,
          "Get a list of peaks in a footprint with a contrast-limited watershed",
          "image"_a, "min_separation"_a, "peak_thresh"_a, "y0"_a, "x0"_a,
          "kappa"_a=0.0, "footprint_thresh"_a=-std::numeric_limits<double>::infinity());

  mod.def("get_footprints", &get_footprints<MatrixF, float>,
          "Create a list of all of the footprints in an image, with their peaks",
          "image"_a, "min_separation"_a, "min_area"_a, "peak_thresh"_a, "footprint_thresh"_a,
          "find_peaks"_a=true, "y0"_a=0, "x0"_a=0, "kappa"_a=0.0);
  mod.def("get_footprints", &get_footprints<MatrixD, double>,
          "Create a list of all of the footprints in an image, with their peaks",
          "image"_a, "min_separation"_a, "min_area"_a, "peak_thresh"_a, "footprint_thresh"_a,
          "find_peaks"_a=true, "y0"_a=0, "x0"_a=0, "kappa"_a=0.0);

  py::class_<Footprint>(mod, "Footprint")
        .def(py::init<MatrixB, std::vector<Peak>, Bounds>(),
             "footprint"_a, "peaks"_a, "bounds"_a)
        .def_property_readonly("data", &Footprint::getFootprint)
        .def_readwrite("peaks", &Footprint::peaks)
        .def_property_readonly("bounds", &Footprint::getBounds)
        .def("add_peak", &Footprint::addPeak);

  py::class_<Peak>(mod, "Peak")
        .def(py::init<int, int, double, double>(),
            "y"_a, "x"_a, "flux"_a, "saddle"_a=std::numeric_limits<double>::quiet_NaN())
        .def_property_readonly("y", &Peak::getY)
        .def_property_readonly("x", &Peak::getX)
        .def_property_readonly("flux", &Peak::getFlux)
        .def_property_readonly("saddle", &Peak::getSaddle);
}
