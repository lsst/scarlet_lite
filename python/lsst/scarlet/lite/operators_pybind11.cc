#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <math.h>
#include <algorithm>
#include <queue>

namespace py = pybind11;

typedef Eigen::Array<int, Eigen::Dynamic, 1> IndexVector;
typedef Eigen::Array<int, 4, 1> Bounds;
typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixB;

template <typename T, typename M>
void new_monotonicity(
    Eigen::Ref<const IndexVector> coord_y,
    Eigen::Ref<const IndexVector> coord_x,
    std::vector<M>& weights,
    Eigen::Ref<M, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> image
){
    const int height = image.rows();
    const int width = image.cols();

    for(int n=0; n<coord_x.size(); n++){
        int px = coord_x[n];
        int py = coord_y[n];

        // Check bounds for 3x3 neighborhood access
        if (py < 0 || py + 2 >= height || px < 0 || px + 2 >= width) {
            throw std::out_of_range("Coordinate (" + std::to_string(py) + ", " +
                                  std::to_string(px) + ") requires 3x3 neighborhood that exceeds image bounds [0, " +
                                  std::to_string(height) + ") x [0, " + std::to_string(width) + ")");
        }

        T ref_flux = 0;
        for(int i=0; i<3; i++){
            for(int j=0; j<3; j++){
                int weight_index = 3*i + j;
                ref_flux += image(py + i, px + j) * weights[weight_index](py, px);
            }
        }
        image(py + 1, px + 1) = std::min(image(py+1, px+1), ref_flux);
    }
}


template <typename T, typename M, typename V>
void prox_weighted_monotonic(
    // Fast implementation of weighted monotonicity constraint
    Eigen::Ref<V> flat_img,
    Eigen::Ref<const M> weights,
    Eigen::Ref<const IndexVector> offsets,
    Eigen::Ref<const IndexVector> dist_idx,
    T const &min_gradient
){
    // Start at the center of the image and set each pixel to the minimum
    // between itself and its reference pixel (which is closer to the peak)
    for(int d=0; d<dist_idx.size(); d++){
        int didx = dist_idx(d);
        T ref_flux = 0;
        for(int i=0; i<offsets.size(); i++){
            if(weights(i,didx)>0){
                int nidx = offsets[i] + didx;
                ref_flux += flat_img(nidx) * weights(i, didx);
            }
        }
        flat_img(didx) = std::min(flat_img(didx), ref_flux*(1-min_gradient));
    }
}

// Apply a 2D filter to an image
template <typename M, typename V>
void apply_filter(
    Eigen::Ref<const M> image,
    Eigen::Ref<const V> values,
    Eigen::Ref<const IndexVector> y_start,
    Eigen::Ref<const IndexVector> y_end,
    Eigen::Ref<const IndexVector> x_start,
    Eigen::Ref<const IndexVector> x_end,
    Eigen::Ref<M, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> result
){
    result.setZero();
    for(int n=0; n<values.size(); n++){
        int rows = image.rows()-y_start(n)-y_end(n);
        int cols = image.cols()-x_start(n)-x_end(n);
        result.block(y_start(n), x_start(n), rows, cols) +=
            values(n) * image.block(y_end(n), x_end(n), rows, cols);
    }
}


// Create a boolean mask for all pixels that are monotonic from  at least one neighbor,
// and create a boolean map of "orphans" that are non-monotonic in all directions.
template <typename M>
void get_valid_monotonic_pixels(
    const int start_i,
    const int start_j,
    Eigen::Ref<M, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> image,
    Eigen::Ref<MatrixB, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> unchecked,
    Eigen::Ref<MatrixB, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> orphans,
    const double variance,
    Eigen::Ref<Bounds, 0, Eigen::Stride<4, 1>> bounds,
    const double thresh=0
) {
    // Create a queue to store pixels that need to be processed
    std::queue<std::pair<int, int>> pixel_queue;
    pixel_queue.push({start_i, start_j});

    // Define directions: down, up, left, right
    const int di[] = {-1, 1, 0, 0};
    const int dj[] = {0, 0, -1, 1};

    // Define image dimensions
    const int nrows = image.rows();
    const int ncols = image.cols();

    while (!pixel_queue.empty()) {
        auto [i, j] = pixel_queue.front();
        pixel_queue.pop();

        const auto image_i_j_var = image(i, j) + variance;

        // Check all four directions
        for (int dir = 0; dir < 4; dir++) {
            int ni = i + di[dir];
            int nj = j + dj[dir];

            // Check bounds
            if ((ni < 0) || (ni >= nrows) || (nj < 0) || (nj >= ncols)) {
                continue;
            }

            const auto image_ni_nj = image(ni, nj);

            // Check if pixel needs to be processed
            if (!unchecked(ni, nj)) {
                continue;
            }

            // Check monotonicity condition
            if (image_ni_nj <= image_i_j_var && (image_ni_nj > thresh)) {
                // Mark as checked and not orphaned
                unchecked(ni, nj) = false;
                orphans(ni, nj) = false;

                // Update bounds
                if (dir == 0 && ni < bounds(0)) {  // down
                    bounds(0) = ni;
                } else if (dir == 1 && ni > bounds(1)) {  // up
                    bounds(1) = ni;
                } else if (dir == 2 && nj < bounds(2)) {  // left
                    bounds(2) = nj;
                } else if (dir == 3 && nj > bounds(3)) {  // right
                    bounds(3) = nj;
                }

                // Add to queue for processing
                pixel_queue.push({ni, nj});
            } else {
                orphans(ni, nj) = true;
            }
        }
    }
}

// Fill in orphans generated by get_valid_monotonic_pixels
template <typename M, typename P>
void linear_interpolate_invalid_pixels(
    Eigen::Ref<const IndexVector> row_indices,
    Eigen::Ref<const IndexVector> column_indices,
    Eigen::Ref<MatrixB, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> unchecked,
    Eigen::Ref<M, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> model,
    Eigen::Ref<MatrixB, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> orphans,
    const double variance,
    bool recursive,
    Eigen::Ref<Bounds, 0, Eigen::Stride<4, 1>> bounds
){
    const int nrows = model.rows();
    const int ncols = model.cols();

    for(int n=0; n<row_indices.size(); n++){
        int i = row_indices(n);
        int j = column_indices(n);

        // Add bounds check for the current pixel
        if (i < 0 || i >= nrows || j < 0 || j >= ncols) {
            throw std::out_of_range("Pixel coordinates (" + std::to_string(i) + ", " +
                                  std::to_string(j) + ") are out of model bounds [0, " +
                                  std::to_string(nrows) + ") x [0, " + std::to_string(ncols) + ")");
        }

        P neighborTotal = 0.0;
        int validNeighbors = 0;
        bool uncheckedNeighbors = false;

        if(!unchecked(i,j)){
            // This pixel has already been updated
            continue;
        }
        // Even if this orphan cannot be updated, we remove it from unchecked
        // so that it isn't attempted again in future iterations.
        unchecked(i,j) = false;

        // Check all of the neighboring pixels with negative gradients and
        // use the maximum value for the interpolation
        if(i < nrows-2 && model(i+2,j) > model(i+1,j)){
            if(unchecked(i+2, j) || unchecked(i+1, j)){
                uncheckedNeighbors = true;
            } else {
                P grad = model(i+2,j) - model(i+1,j);
                neighborTotal += model(i+1,j)-grad;
                validNeighbors += 1;
            }
        }
        if(i >= 2 && model(i-2,j) > model(i-1,j)){
            if(unchecked(i-2, j) || unchecked(i-1, j)){
                uncheckedNeighbors = true;
            } else {
                P grad = model(i-2,j) - model(i-1,j);
                neighborTotal += model(i-1,j)-grad;
                validNeighbors += 1;
            }
        }
        if(j < ncols-2 && model(i,j+2) > model(i,j+1)){
            if(unchecked(i,j+2) || unchecked(i,j+1)){
                uncheckedNeighbors = true;
            } else {
                P grad = model(i,j+2) - model(i,j+1);
                neighborTotal += model(i,j+1)-grad;
                validNeighbors += 1;
            }
        }
        if(j >= 2 && model(i,j-2) > model(i,j-1)){
            if(unchecked(i, j-2) || unchecked(i,j-1)){
                uncheckedNeighbors = true;
            } else {
                P grad = model(i,j-2) - model(i,j-1);
                neighborTotal += model(i,j-1)-grad;
                validNeighbors += 1;
            }
        }
        // If the non-monotonic pixel was updated then update the
        // model with the interpolated value and search for more monotonic pixels
        if(neighborTotal > 0){
            // Update the model and orphan status _before_ checking neighbors
            model(i,j) = neighborTotal / validNeighbors;
            orphans(i,j) = false;

            if(i < bounds(0)){
                bounds(0) = i;
            } else if(i > bounds(1)){
                bounds(1) = i;
            }
            if(j < bounds(2)){
                bounds(2) = j;
            } else if(j > bounds(3)){
                bounds(3) = j;
            }
            if(recursive){
                get_valid_monotonic_pixels(i, j, model, unchecked, orphans, variance, bounds);
            } else {
                if(i > 0 && unchecked(i-1,j)){
                    orphans(i-1,j) = true;
                }
                if(i < nrows-1 && unchecked(i+1,j)){
                    orphans(i+1,j) = true;
                }
                if(j > 0 && unchecked(i,j-1)){
                    orphans(i,j-1) = true;
                }
                if(j < ncols-1 && unchecked(i,j+1)){
                    orphans(i,j+1) = true;
                }
            }
        } else if(uncheckedNeighbors){
            unchecked(i,j) = false;
        } else {
            // This is still an orphan, but we checked it so it won't be iterated on again.
            orphans(i,j) = true;
            model(i,j) = 0;
        }
    }
}

PYBIND11_MODULE(operators_pybind11, mod)
{
  mod.doc() = "operators_pybind11", "Fast proximal operators";

  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixF;
  typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VectorF;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixD;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorD;

  mod.def("new_monotonicity", &new_monotonicity<float, MatrixF>, "Weighted Monotonic Proximal Operator");
  mod.def("new_monotonicity", &new_monotonicity<double, MatrixD>, "Weighted Monotonic Proximal Operator");

  mod.def("prox_weighted_monotonic", &prox_weighted_monotonic<float, MatrixF, VectorF>,
          "Weighted Monotonic Proximal Operator");
  mod.def("prox_weighted_monotonic", &prox_weighted_monotonic<double, MatrixD, VectorD>,
          "Weighted Monotonic Proximal Operator");

  mod.def("apply_filter", &apply_filter<MatrixF, VectorF>, "Apply a filter to a 2D image");
  mod.def("apply_filter", &apply_filter<MatrixD, VectorD>, "Apply a filter to a 2D image");

  mod.def("get_valid_monotonic_pixels", &get_valid_monotonic_pixels<MatrixF>,
          "Create a boolean mask for pixels that are monotonic from the center along some path");
  mod.def("get_valid_monotonic_pixels", &get_valid_monotonic_pixels<MatrixD>,
          "Create a boolean mask for pixels that are monotonic from the center along some path");

  mod.def("linear_interpolate_invalid_pixels", &linear_interpolate_invalid_pixels<MatrixF, float>,
          "Fill in non-monotonic pixels by interpolating based on the gradients of its neighbors");
  mod.def("linear_interpolate_invalid_pixels", &linear_interpolate_invalid_pixels<MatrixD, double>,
          "Fill in non-monotonic pixels by interpolating based on the gradients of its neighbors");
}
