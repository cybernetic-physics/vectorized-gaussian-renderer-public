#include "debug_ops.h"

#include <climits>
#include <math.h>

namespace flashgs_debug {
namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

enum ProjectionRejection : int {
  kProjectionAccepted = 0,
  kInvalidGaussianId = 1,
  kOpacityRejected = 2,
  kDepthRejected = 3,
  kCovarianceRejected = 4,
  kRadiusRejected = 5,
  kScreenBoundsRejected = 6,
  kEmptyTileRectRejected = 7,
};

enum PixelBranch : int {
  kPixelNotEvaluated = 0,
  kPixelPositivePower = 1,
  kPixelLowPower = 2,
  kPixelLowAlpha = 3,
  kPixelContributor = 4,
};

enum CompositorBranch : int {
  kCompositorAbsent = 0,
  kCompositorPositivePower = 1,
  kCompositorLowPower = 2,
  kCompositorLowAlpha = 3,
  kCompositorDoneBeforeCandidate = 4,
  kCompositorTransmittanceCutoff = 5,
  kCompositorContributed = 6,
};

enum FloatField : int {
  kOpacityRaw = 0,
  kCameraX = 1,
  kCameraY = 2,
  kCameraZ = 3,
  kPointX = 4,
  kPointY = 5,
  kCovariance00 = 6,
  kCovariance01 = 7,
  kCovariance11 = 8,
  kDeterminant = 9,
  kOpacityExtend = 10,
  kExtend = 11,
  kHalfSupportSquared = 12,
  kConicX = 13,
  kConicY = 14,
  kConicZ = 15,
  kTilePixelMinX = 16,
  kTilePixelMinY = 17,
  kTilePixelMaxX = 18,
  kTilePixelMaxY = 19,
  kFirstDx = 20,
  kFirstA = 21,
  kFirstB = 22,
  kFirstC = 23,
  kFirstDelta = 24,
  kFirstT1 = 25,
  kFirstT2 = 26,
  kSecondDy = 27,
  kSecondA = 28,
  kSecondB = 29,
  kSecondC = 30,
  kSecondDelta = 31,
  kSecondT1 = 32,
  kSecondT2 = 33,
  kTileCenterX = 34,
  kTileCenterY = 35,
  kTileCenterQ = 36,
  kCorner00Q = 37,
  kCorner10Q = 38,
  kCorner01Q = 39,
  kCorner11Q = 40,
  kTargetPowerReprojected = 41,
  kTargetAlphaReprojected = 42,
  kWorkspacePointX = 43,
  kWorkspacePointY = 44,
  kWorkspaceConicX = 45,
  kWorkspaceConicY = 46,
  kWorkspaceConicZ = 47,
  kWorkspaceOpacity = 48,
  kCompositorPower = 49,
  kCompositorAlpha = 50,
  kCompositorPreTransmittance = 51,
  kCompositorNextTransmittance = 52,
  kCompositorWeight = 53,
  kCompositorLoadedPointX = 54,
  kCompositorLoadedPointY = 55,
  kCompositorLoadedConicX = 56,
  kCompositorLoadedConicY = 57,
  kCompositorLoadedConicZ = 58,
  kCompositorLoadedOpacity = 59,
};

enum IntField : int {
  kGaussianId = 0,
  kGaussianIdValid = 1,
  kProjectionRejection = 2,
  kRadiusX = 3,
  kRadiusY = 4,
  kRectMinX = 5,
  kRectMinY = 6,
  kRectMaxX = 7,
  kRectMaxY = 8,
  kTargetTileX = 9,
  kTargetTileY = 10,
  kTargetTileId = 11,
  kTargetTileInRect = 12,
  kOldContainsCenter = 13,
  kOldFirstSegmentEvaluated = 14,
  kOldFirstSegmentResult = 15,
  kOldSecondSegmentEvaluated = 16,
  kOldSecondSegmentResult = 17,
  kOldPredicateAccept = 18,
  kTileCenterInsideEllipse = 19,
  kAllTileCornersInsideEllipse = 20,
  kTargetPixelBranch = 21,
  kUnsortedAnyCount = 22,
  kUnsortedTargetCount = 23,
  kSortedAnyCount = 24,
  kSortedTargetCount = 25,
  kRangeStart = 26,
  kRangeEnd = 27,
  kCandidateInRangeCount = 28,
  kSortedPosition = 29,
  kCompositorSeen = 30,
  kCompositorBranch = 31,
  kCompositorFeatureGaussianId = 32,
  kCompositorFeatureMatchesSortedId = 33,
  kCompositorInitialPairSlot = 34,
  kCompositorOffsetAssignmentGuard = 35,
  kCompositorLoadEnableGuard = 36,
  kCompositorZeroOffsetFallback = 37,
  kCompositorFeatureLoadOffset = 38,
};

struct Projection {
  int rejection = kProjectionAccepted;
  float opacity_raw = NAN;
  float opacity = NAN;
  float camera_x = NAN;
  float camera_y = NAN;
  float camera_z = NAN;
  float2 point = {NAN, NAN};
  float covariance_00 = NAN;
  float covariance_01 = NAN;
  float covariance_11 = NAN;
  float determinant = NAN;
  float opacity_extend = NAN;
  float extend = NAN;
  float half_support_squared = NAN;
  float3 conic = {NAN, NAN, NAN};
  int radius_x = 0;
  int radius_y = 0;
  int2 rect_min = {0, 0};
  int2 rect_max = {0, 0};
};

struct Predicate {
  float2 pixel_min = {NAN, NAN};
  float2 pixel_max = {NAN, NAN};
  float first_dx = NAN;
  float first_a = NAN;
  float first_b = NAN;
  float first_c = NAN;
  float first_delta = NAN;
  float first_t1 = NAN;
  float first_t2 = NAN;
  float second_dy = NAN;
  float second_a = NAN;
  float second_b = NAN;
  float second_c = NAN;
  float second_delta = NAN;
  float second_t1 = NAN;
  float second_t2 = NAN;
  float2 tile_center = {NAN, NAN};
  float tile_center_q = NAN;
  float corner_q[4] = {NAN, NAN, NAN, NAN};
  bool contains_center = false;
  bool first_evaluated = false;
  bool first_result = false;
  bool second_evaluated = false;
  bool second_result = false;
  bool old_accept = false;
  bool tile_center_inside = false;
  bool all_corners_inside = false;
};

__device__ __forceinline__ int clamp_int(int value, int lower, int upper) {
  return min(upper, max(lower, value));
}

__device__ __forceinline__ void get_rect(
    float2 center,
    int radius_x,
    int radius_y,
    int2* rect_min,
    int2* rect_max,
    int grid_x,
    int grid_y) {
  rect_min->x = clamp_int(
      static_cast<int>((center.x - radius_x) / kBlockX), 0, grid_x);
  rect_min->y = clamp_int(
      static_cast<int>((center.y - radius_y) / kBlockY), 0, grid_y);
  rect_max->x = clamp_int(
      static_cast<int>((center.x + radius_x) / kBlockX) + 1, 0, grid_x);
  rect_max->y = clamp_int(
      static_cast<int>((center.y + radius_y) / kBlockY) + 1, 0, grid_y);
}

__device__ __forceinline__ Projection project_gaussian(
    int gaussian,
    const float* positions,
    const flashgs_adapter::cov3d_t* covariances,
    const float* opacities,
    const float* view,
    const float* intrinsic,
    int width,
    int height,
    float near_plane,
    float far_plane,
    float support_sigma,
    float covariance_epsilon) {
  Projection result;
  result.opacity_raw = opacities[gaussian];
  if (!(result.opacity_raw >= FLASHGS_ALPHA_THRESHOLD)) {
    result.rejection = kOpacityRejected;
    return result;
  }
  const float* mean = positions + static_cast<int64_t>(gaussian) * 3;
  result.camera_x =
      view[0] * mean[0] + view[1] * mean[1] + view[2] * mean[2] + view[3];
  result.camera_y =
      view[4] * mean[0] + view[5] * mean[1] + view[6] * mean[2] + view[7];
  result.camera_z =
      view[8] * mean[0] + view[9] * mean[1] + view[10] * mean[2] + view[11];
  if (result.camera_z < near_plane || result.camera_z > far_plane) {
    result.rejection = kDepthRejected;
    return result;
  }

  const float focal_x = intrinsic[0];
  const float focal_y = intrinsic[4];
  const float center_x = intrinsic[2];
  const float center_y = intrinsic[5];
  const float inverse_z = 1.0f / result.camera_z;
  result.point.x = focal_x * result.camera_x * inverse_z + center_x;
  result.point.y = focal_y * result.camera_y * inverse_z + center_y;

  const flashgs_adapter::cov3d_t object = covariances[gaussian];
  const float r00 = view[0];
  const float r01 = view[1];
  const float r02 = view[2];
  const float r10 = view[4];
  const float r11 = view[5];
  const float r12 = view[6];
  const float r20 = view[8];
  const float r21 = view[9];
  const float r22 = view[10];
  const float c00 = object.s[0];
  const float c01 = object.s[1];
  const float c02 = object.s[2];
  const float c11 = object.s[3];
  const float c12 = object.s[4];
  const float c22 = object.s[5];
#define ROTATED_COV(A0, A1, A2, B0, B1, B2) \
  ((A0) * ((B0) * c00 + (B1) * c01 + (B2) * c02) + \
   (A1) * ((B0) * c01 + (B1) * c11 + (B2) * c12) + \
   (A2) * ((B0) * c02 + (B1) * c12 + (B2) * c22))
  const float camera_cov_00 = ROTATED_COV(r00, r01, r02, r00, r01, r02);
  const float camera_cov_01 = ROTATED_COV(r00, r01, r02, r10, r11, r12);
  const float camera_cov_02 = ROTATED_COV(r00, r01, r02, r20, r21, r22);
  const float camera_cov_11 = ROTATED_COV(r10, r11, r12, r10, r11, r12);
  const float camera_cov_12 = ROTATED_COV(r10, r11, r12, r20, r21, r22);
  const float camera_cov_22 = ROTATED_COV(r20, r21, r22, r20, r21, r22);
#undef ROTATED_COV

  const float tan_fov_x = 0.5f * width / focal_x;
  const float tan_fov_y = 0.5f * height / focal_y;
  const float limit_x_positive =
      (width - center_x) / focal_x + 0.3f * tan_fov_x;
  const float limit_x_negative = center_x / focal_x + 0.3f * tan_fov_x;
  const float limit_y_positive =
      (height - center_y) / focal_y + 0.3f * tan_fov_y;
  const float limit_y_negative = center_y / focal_y + 0.3f * tan_fov_y;
  const float clamped_x = result.camera_z * fminf(
      limit_x_positive,
      fmaxf(-limit_x_negative, result.camera_x * inverse_z));
  const float clamped_y = result.camera_z * fminf(
      limit_y_positive,
      fmaxf(-limit_y_negative, result.camera_y * inverse_z));
  const float inverse_z_squared = inverse_z * inverse_z;
  const float j00 = focal_x * inverse_z;
  const float j02 = -focal_x * clamped_x * inverse_z_squared;
  const float j11 = focal_y * inverse_z;
  const float j12 = -focal_y * clamped_y * inverse_z_squared;
  result.covariance_00 =
      j00 * j00 * camera_cov_00 +
      2.0f * j00 * j02 * camera_cov_02 +
      j02 * j02 * camera_cov_22 + covariance_epsilon;
  result.covariance_01 =
      j00 * j11 * camera_cov_01 +
      j00 * j12 * camera_cov_02 +
      j02 * j11 * camera_cov_12 +
      j02 * j12 * camera_cov_22;
  result.covariance_11 =
      j11 * j11 * camera_cov_11 +
      2.0f * j11 * j12 * camera_cov_12 +
      j12 * j12 * camera_cov_22 + covariance_epsilon;
  result.determinant =
      result.covariance_00 * result.covariance_11 -
      result.covariance_01 * result.covariance_01;
  if (!(result.determinant > 0.0f)) {
    result.rejection = kCovarianceRejected;
    return result;
  }
  result.opacity_extend = sqrtf(
      2.0f * __logf(result.opacity_raw / FLASHGS_ALPHA_THRESHOLD));
  result.extend = fminf(support_sigma, result.opacity_extend);
  result.half_support_squared = 0.5f * result.extend * result.extend;
  result.radius_x = static_cast<int>(
      ceilf(result.extend * sqrtf(result.covariance_00)));
  result.radius_y = static_cast<int>(
      ceilf(result.extend * sqrtf(result.covariance_11)));
  if (result.radius_x <= 0 && result.radius_y <= 0) {
    result.rejection = kRadiusRejected;
    return result;
  }
  if (
      result.point.x + result.radius_x <= 0.0f ||
      result.point.y + result.radius_y <= 0.0f ||
      result.point.x - result.radius_x >= width ||
      result.point.y - result.radius_y >= height) {
    result.rejection = kScreenBoundsRejected;
    return result;
  }
  const float inverse_determinant = 1.0f / result.determinant;
  result.conic.x = result.covariance_11 * inverse_determinant;
  result.conic.y = -result.covariance_01 * inverse_determinant;
  result.conic.z = result.covariance_00 * inverse_determinant;
  result.opacity = fminf(fmaxf(result.opacity_raw, 0.0f), 1.0f);
  const int grid_x = (width + kBlockX - 1) / kBlockX;
  const int grid_y = (height + kBlockY - 1) / kBlockY;
  get_rect(
      result.point,
      result.radius_x,
      result.radius_y,
      &result.rect_min,
      &result.rect_max,
      grid_x,
      grid_y);
  if (
      (result.rect_max.x - result.rect_min.x) *
          (result.rect_max.y - result.rect_min.y) <= 0) {
    result.rejection = kEmptyTileRectRejected;
  }
  return result;
}

__device__ __forceinline__ bool segment_result(
    float a,
    float b,
    float c,
    float center,
    float lower,
    float upper,
    float* delta,
    float* t1,
    float* t2) {
  *delta = b * b - 4.0f * a * c;
  *t1 = (lower - center) * (2.0f * a) + b;
  *t2 = (upper - center) * (2.0f * a) + b;
  return *delta >= 0.0f &&
      (*t1 <= 0.0f || *t1 * *t1 <= *delta) &&
      (*t2 >= 0.0f || *t2 * *t2 <= *delta);
}

__device__ __forceinline__ float ellipse_q(
    float2 point,
    float2 center,
    float3 conic) {
  const float dx = point.x - center.x;
  const float dy = point.y - center.y;
  return conic.x * dx * dx + 2.0f * conic.y * dx * dy +
      conic.z * dy * dy;
}

__device__ __forceinline__ Predicate evaluate_old_predicate(
    const Projection& projection,
    int tile_x,
    int tile_y,
    int width,
    int height) {
  Predicate result;
  result.pixel_min = {
      tile_x * kBlockX + 0.5f,
      tile_y * kBlockY + 0.5f,
  };
  result.pixel_max = {
      fminf(width - 0.5f, result.pixel_min.x + kBlockX - 1),
      fminf(height - 0.5f, result.pixel_min.y + kBlockY - 1),
  };
  result.contains_center =
      projection.point.x >= result.pixel_min.x &&
      projection.point.x <= result.pixel_max.x &&
      projection.point.y >= result.pixel_min.y &&
      projection.point.y <= result.pixel_max.y;

  result.first_dx =
      projection.point.x * 2.0f < result.pixel_min.x + result.pixel_max.x
      ? projection.point.x - result.pixel_min.x
      : projection.point.x - result.pixel_max.x;
  result.first_a = projection.conic.z;
  result.first_b = -2.0f * projection.conic.y * result.first_dx;
  result.first_c =
      projection.conic.x * result.first_dx * result.first_dx -
      2.0f * projection.half_support_squared;
  result.first_result = segment_result(
      result.first_a,
      result.first_b,
      result.first_c,
      projection.point.y,
      result.pixel_min.y,
      result.pixel_max.y,
      &result.first_delta,
      &result.first_t1,
      &result.first_t2);

  result.second_dy =
      projection.point.y * 2.0f < result.pixel_min.y + result.pixel_max.y
      ? projection.point.y - result.pixel_min.y
      : projection.point.y - result.pixel_max.y;
  result.second_a = projection.conic.x;
  result.second_b = -2.0f * projection.conic.y * result.second_dy;
  result.second_c =
      projection.conic.z * result.second_dy * result.second_dy -
      2.0f * projection.half_support_squared;
  result.second_result = segment_result(
      result.second_a,
      result.second_b,
      result.second_c,
      projection.point.x,
      result.pixel_min.x,
      result.pixel_max.x,
      &result.second_delta,
      &result.second_t1,
      &result.second_t2);

  // These flags preserve the exact short-circuit path of
  // contains_center || block_intersects_ellipse(first || second).
  result.first_evaluated = !result.contains_center;
  result.second_evaluated = result.first_evaluated && !result.first_result;
  result.old_accept = result.contains_center ||
      result.first_result || result.second_result;

  result.tile_center = {
      0.5f * (result.pixel_min.x + result.pixel_max.x),
      0.5f * (result.pixel_min.y + result.pixel_max.y),
  };
  result.tile_center_q = ellipse_q(
      result.tile_center, projection.point, projection.conic);
  result.corner_q[0] = ellipse_q(
      result.pixel_min, projection.point, projection.conic);
  result.corner_q[1] = ellipse_q(
      {result.pixel_max.x, result.pixel_min.y},
      projection.point,
      projection.conic);
  result.corner_q[2] = ellipse_q(
      {result.pixel_min.x, result.pixel_max.y},
      projection.point,
      projection.conic);
  result.corner_q[3] = ellipse_q(
      result.pixel_max, projection.point, projection.conic);
  const float limit = 2.0f * projection.half_support_squared;
  result.tile_center_inside = result.tile_center_q <= limit;
  result.all_corners_inside =
      result.corner_q[0] <= limit && result.corner_q[1] <= limit &&
      result.corner_q[2] <= limit && result.corner_q[3] <= limit;
  return result;
}

__device__ __forceinline__ int evaluate_target_pixel(
    const Projection& projection,
    int pixel_x,
    int pixel_y,
    float* power,
    float* alpha) {
  const float dx = projection.point.x - (pixel_x + 0.5f);
  const float dy = projection.point.y - (pixel_y + 0.5f);
  *power = -0.5f * (
      projection.conic.x * dx * dx +
      2.0f * projection.conic.y * dx * dy +
      projection.conic.z * dy * dy);
  *alpha = NAN;
  if (*power > 0.0f) {
    return kPixelPositivePower;
  }
  if (*power < -20.0f) {
    return kPixelLowPower;
  }
  *alpha = fminf(FLASHGS_MAX_ALPHA, projection.opacity * __expf(*power));
  if (*alpha < FLASHGS_ALPHA_THRESHOLD) {
    return kPixelLowAlpha;
  }
  return kPixelContributor;
}

__global__ void score_contributors_kernel(
    int count,
    const float* positions,
    const flashgs_adapter::cov3d_t* covariances,
    const float* opacities,
    const int64_t* semantic_ids,
    const float* view,
    const float* intrinsic,
    int width,
    int height,
    float near_plane,
    float far_plane,
    float support_sigma,
    float covariance_epsilon,
    int target_pixel_x,
    int target_pixel_y,
    int64_t expected_semantic_id,
    bool require_old_predicate_rejection,
    float* scores) {
  const int gaussian = blockIdx.x * blockDim.x + threadIdx.x;
  if (gaussian >= count) {
    return;
  }
  if (
      expected_semantic_id >= 0 &&
      semantic_ids[gaussian] != expected_semantic_id) {
    return;
  }
  const Projection projection = project_gaussian(
      gaussian,
      positions,
      covariances,
      opacities,
      view,
      intrinsic,
      width,
      height,
      near_plane,
      far_plane,
      support_sigma,
      covariance_epsilon);
  if (projection.rejection != kProjectionAccepted) {
    return;
  }
  float power = NAN;
  float alpha = NAN;
  if (
      evaluate_target_pixel(
          projection,
          target_pixel_x,
          target_pixel_y,
          &power,
          &alpha) != kPixelContributor) {
    return;
  }
  if (require_old_predicate_rejection) {
    const int tile_x = target_pixel_x / kBlockX;
    const int tile_y = target_pixel_y / kBlockY;
    const bool tile_in_rect =
        tile_x >= projection.rect_min.x && tile_x < projection.rect_max.x &&
        tile_y >= projection.rect_min.y && tile_y < projection.rect_max.y;
    if (!tile_in_rect) {
      return;
    }
    const Predicate predicate = evaluate_old_predicate(
        projection, tile_x, tile_y, width, height);
    if (predicate.old_accept) {
      return;
    }
  }
  scores[gaussian] = alpha;
}

__device__ __forceinline__ void write_projection_trace(
    const Projection& projection,
    const Predicate& predicate,
    int target_tile_x,
    int target_tile_y,
    int target_tile_id,
    bool target_tile_in_rect,
    int target_pixel_branch,
    float target_power,
    float target_alpha,
    float* floats,
    int64_t* ints) {
  floats[kOpacityRaw] = projection.opacity_raw;
  floats[kCameraX] = projection.camera_x;
  floats[kCameraY] = projection.camera_y;
  floats[kCameraZ] = projection.camera_z;
  floats[kPointX] = projection.point.x;
  floats[kPointY] = projection.point.y;
  floats[kCovariance00] = projection.covariance_00;
  floats[kCovariance01] = projection.covariance_01;
  floats[kCovariance11] = projection.covariance_11;
  floats[kDeterminant] = projection.determinant;
  floats[kOpacityExtend] = projection.opacity_extend;
  floats[kExtend] = projection.extend;
  floats[kHalfSupportSquared] = projection.half_support_squared;
  floats[kConicX] = projection.conic.x;
  floats[kConicY] = projection.conic.y;
  floats[kConicZ] = projection.conic.z;
  floats[kTilePixelMinX] = predicate.pixel_min.x;
  floats[kTilePixelMinY] = predicate.pixel_min.y;
  floats[kTilePixelMaxX] = predicate.pixel_max.x;
  floats[kTilePixelMaxY] = predicate.pixel_max.y;
  floats[kFirstDx] = predicate.first_dx;
  floats[kFirstA] = predicate.first_a;
  floats[kFirstB] = predicate.first_b;
  floats[kFirstC] = predicate.first_c;
  floats[kFirstDelta] = predicate.first_delta;
  floats[kFirstT1] = predicate.first_t1;
  floats[kFirstT2] = predicate.first_t2;
  floats[kSecondDy] = predicate.second_dy;
  floats[kSecondA] = predicate.second_a;
  floats[kSecondB] = predicate.second_b;
  floats[kSecondC] = predicate.second_c;
  floats[kSecondDelta] = predicate.second_delta;
  floats[kSecondT1] = predicate.second_t1;
  floats[kSecondT2] = predicate.second_t2;
  floats[kTileCenterX] = predicate.tile_center.x;
  floats[kTileCenterY] = predicate.tile_center.y;
  floats[kTileCenterQ] = predicate.tile_center_q;
  floats[kCorner00Q] = predicate.corner_q[0];
  floats[kCorner10Q] = predicate.corner_q[1];
  floats[kCorner01Q] = predicate.corner_q[2];
  floats[kCorner11Q] = predicate.corner_q[3];
  floats[kTargetPowerReprojected] = target_power;
  floats[kTargetAlphaReprojected] = target_alpha;
  ints[kProjectionRejection] = projection.rejection;
  ints[kRadiusX] = projection.radius_x;
  ints[kRadiusY] = projection.radius_y;
  ints[kRectMinX] = projection.rect_min.x;
  ints[kRectMinY] = projection.rect_min.y;
  ints[kRectMaxX] = projection.rect_max.x;
  ints[kRectMaxY] = projection.rect_max.y;
  ints[kTargetTileX] = target_tile_x;
  ints[kTargetTileY] = target_tile_y;
  ints[kTargetTileId] = target_tile_id;
  ints[kTargetTileInRect] = target_tile_in_rect;
  ints[kOldContainsCenter] = predicate.contains_center;
  ints[kOldFirstSegmentEvaluated] = predicate.first_evaluated;
  ints[kOldFirstSegmentResult] = predicate.first_result;
  ints[kOldSecondSegmentEvaluated] = predicate.second_evaluated;
  ints[kOldSecondSegmentResult] = predicate.second_result;
  ints[kOldPredicateAccept] = predicate.old_accept;
  ints[kTileCenterInsideEllipse] = predicate.tile_center_inside;
  ints[kAllTileCornersInsideEllipse] = predicate.all_corners_inside;
  ints[kTargetPixelBranch] = target_pixel_branch;
}

__global__ void trace_kernel(
    int count,
    const float* positions,
    const flashgs_adapter::cov3d_t* covariances,
    const float* opacities,
    const float* view,
    const float* intrinsic,
    int width,
    int height,
    float near_plane,
    float far_plane,
    float support_sigma,
    float covariance_epsilon,
    int target_pixel_x,
    int target_pixel_y,
    const uint64_t* keys_unsorted,
    const uint32_t* values_unsorted,
    const uint64_t* keys_sorted,
    const uint32_t* values_sorted,
    const int2* ranges,
    const float2* points_xy,
    const float4* conic_opacity,
    int capacity,
    const int64_t* candidate_ids,
    int candidate_count,
    float* float_trace,
    int64_t* int_trace) {
  const int candidate_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate_index >= candidate_count) {
    return;
  }
  float* floats = float_trace +
      static_cast<int64_t>(candidate_index) * kFloatFieldCount;
  int64_t* ints = int_trace +
      static_cast<int64_t>(candidate_index) * kIntFieldCount;
  for (int field = 0; field < kFloatFieldCount; ++field) {
    floats[field] = NAN;
  }
  for (int field = 0; field < kIntFieldCount; ++field) {
    ints[field] = 0;
  }
  const int64_t candidate64 = candidate_ids[candidate_index];
  ints[kGaussianId] = candidate64;
  ints[kSortedPosition] = -1;
  ints[kCompositorFeatureGaussianId] = -1;
  ints[kCompositorInitialPairSlot] = -1;
  ints[kCompositorOffsetAssignmentGuard] = -1;
  ints[kCompositorLoadEnableGuard] = -1;
  ints[kCompositorFeatureLoadOffset] = -1;
  if (candidate64 < 0 || candidate64 >= count) {
    ints[kProjectionRejection] = kInvalidGaussianId;
    return;
  }
  const int candidate = static_cast<int>(candidate64);
  ints[kGaussianIdValid] = 1;
  const int grid_x = (width + kBlockX - 1) / kBlockX;
  const int grid_y = (height + kBlockY - 1) / kBlockY;
  const int tile_count = grid_x * grid_y;
  const int target_tile_x = target_pixel_x / kBlockX;
  const int target_tile_y = target_pixel_y / kBlockY;
  const int target_tile_id = target_tile_y * grid_x + target_tile_x;
  const Projection projection = project_gaussian(
      candidate,
      positions,
      covariances,
      opacities,
      view,
      intrinsic,
      width,
      height,
      near_plane,
      far_plane,
      support_sigma,
      covariance_epsilon);
  Predicate predicate;
  bool target_tile_in_rect = false;
  int target_pixel_branch = kPixelNotEvaluated;
  float target_power = NAN;
  float target_alpha = NAN;
  if (projection.rejection == kProjectionAccepted) {
    target_tile_in_rect =
        target_tile_x >= projection.rect_min.x &&
        target_tile_x < projection.rect_max.x &&
        target_tile_y >= projection.rect_min.y &&
        target_tile_y < projection.rect_max.y;
    predicate = evaluate_old_predicate(
        projection, target_tile_x, target_tile_y, width, height);
    target_pixel_branch = evaluate_target_pixel(
        projection,
        target_pixel_x,
        target_pixel_y,
        &target_power,
        &target_alpha);
  }
  write_projection_trace(
      projection,
      predicate,
      target_tile_x,
      target_tile_y,
      target_tile_id,
      target_tile_in_rect,
      target_pixel_branch,
      target_power,
      target_alpha,
      floats,
      ints);

  for (int index = 0; index < capacity; ++index) {
    const uint32_t tile = static_cast<uint32_t>(keys_unsorted[index] >> 32);
    if (tile < static_cast<uint32_t>(tile_count) &&
        values_unsorted[index] == static_cast<uint32_t>(candidate)) {
      ++ints[kUnsortedAnyCount];
      if (tile == static_cast<uint32_t>(target_tile_id)) {
        ++ints[kUnsortedTargetCount];
      }
    }
  }
  for (int index = 0; index < capacity; ++index) {
    const uint32_t tile = static_cast<uint32_t>(keys_sorted[index] >> 32);
    if (tile < static_cast<uint32_t>(tile_count) &&
        values_sorted[index] == static_cast<uint32_t>(candidate)) {
      ++ints[kSortedAnyCount];
      if (tile == static_cast<uint32_t>(target_tile_id)) {
        ++ints[kSortedTargetCount];
      }
    }
  }
  if (ints[kUnsortedAnyCount] > 0) {
    const float2 workspace_point = points_xy[candidate];
    const float4 workspace_conic = conic_opacity[candidate];
    floats[kWorkspacePointX] = workspace_point.x;
    floats[kWorkspacePointY] = workspace_point.y;
    floats[kWorkspaceConicX] = workspace_conic.x;
    floats[kWorkspaceConicY] = workspace_conic.y;
    floats[kWorkspaceConicZ] = workspace_conic.z;
    floats[kWorkspaceOpacity] = workspace_conic.w;
  }

  const int2 raw_range = ranges[target_tile_id];
  ints[kRangeStart] = raw_range.x;
  ints[kRangeEnd] = raw_range.y;
  const int range_start = clamp_int(raw_range.x, 0, capacity);
  const int range_end = clamp_int(raw_range.y, range_start, capacity);
  for (int index = range_start; index < range_end; ++index) {
    if (values_sorted[index] == static_cast<uint32_t>(candidate)) {
      ++ints[kCandidateInRangeCount];
      if (ints[kSortedPosition] < 0) {
        ints[kSortedPosition] = index;
      }
    }
  }

  float transmittance = 1.0f;
  bool terminated = false;
  for (int index = range_start; index < range_end; ++index) {
    const uint32_t gaussian = values_sorted[index];
    // Reproduce the repaired adapter's optimized cooperative-load schedule.
    // Its initial offset-assignment guards now exactly match the corresponding
    // load-enable guards: +0 for slot 0 and +1 for slot 1.  Keep this
    // source-grounded model bound to render.cu through
    // flashgs_debug_native_loader.py's production hashes.
    const int relative_index = index - range_start;
    const bool initial_pair_slot = relative_index == 0 || relative_index == 1;
    const bool offset_assignment_guard = relative_index == 0
        ? range_start < range_end
        : (relative_index == 1 ? range_start + 1 < range_end : true);
    const bool load_enable_guard = relative_index == 0
        ? range_start < range_end
        : (relative_index == 1 ? range_start + 1 < range_end : true);
    const bool zero_offset_fallback = initial_pair_slot &&
        !offset_assignment_guard && load_enable_guard;
    const uint32_t feature_gaussian = zero_offset_fallback ? 0u : gaussian;
    const float2 point = points_xy[feature_gaussian];
    const float4 conic = conic_opacity[feature_gaussian];
    const float dx = point.x - (target_pixel_x + 0.5f);
    const float dy = point.y - (target_pixel_y + 0.5f);
    const float power = -0.5f * (
        conic.x * dx * dx + 2.0f * conic.y * dx * dy +
        conic.z * dy * dy);
    const bool is_candidate = gaussian == static_cast<uint32_t>(candidate);
    if (is_candidate) {
      ints[kCompositorSeen] = 1;
      ints[kCompositorFeatureGaussianId] = feature_gaussian;
      ints[kCompositorFeatureLoadOffset] = feature_gaussian;
      ints[kCompositorFeatureMatchesSortedId] =
          feature_gaussian == gaussian;
      ints[kCompositorInitialPairSlot] =
          initial_pair_slot ? relative_index : -1;
      ints[kCompositorOffsetAssignmentGuard] =
          initial_pair_slot ? offset_assignment_guard : -1;
      ints[kCompositorLoadEnableGuard] =
          initial_pair_slot ? load_enable_guard : -1;
      ints[kCompositorZeroOffsetFallback] = zero_offset_fallback;
      floats[kCompositorPower] = power;
      floats[kCompositorLoadedPointX] = point.x;
      floats[kCompositorLoadedPointY] = point.y;
      floats[kCompositorLoadedConicX] = conic.x;
      floats[kCompositorLoadedConicY] = conic.y;
      floats[kCompositorLoadedConicZ] = conic.z;
      floats[kCompositorLoadedOpacity] = conic.w;
      floats[kCompositorPreTransmittance] = transmittance;
    }
    if (power > 0.0f) {
      if (is_candidate) {
        ints[kCompositorBranch] = kCompositorPositivePower;
      }
      continue;
    }
    if (power < -20.0f) {
      if (is_candidate) {
        ints[kCompositorBranch] = kCompositorLowPower;
      }
      continue;
    }
    const float alpha = fminf(FLASHGS_MAX_ALPHA, conic.w * __expf(power));
    if (is_candidate) {
      floats[kCompositorAlpha] = alpha;
    }
    if (alpha < FLASHGS_ALPHA_THRESHOLD) {
      if (is_candidate) {
        ints[kCompositorBranch] = kCompositorLowAlpha;
      }
      continue;
    }
    const float next_transmittance = transmittance * (1.0f - alpha);
    if (is_candidate) {
      floats[kCompositorNextTransmittance] = next_transmittance;
    }
    if (next_transmittance <= FLASHGS_TRANSMITTANCE_THRESHOLD) {
      if (is_candidate) {
        ints[kCompositorBranch] = kCompositorTransmittanceCutoff;
      }
      terminated = true;
      break;
    }
    if (is_candidate) {
      floats[kCompositorWeight] = transmittance * alpha;
      ints[kCompositorBranch] = kCompositorContributed;
    }
    transmittance = next_transmittance;
  }
  if (
      terminated && ints[kCandidateInRangeCount] > 0 &&
      !ints[kCompositorSeen]) {
    ints[kCompositorBranch] = kCompositorDoneBeforeCandidate;
  }
}

}  // namespace

void score_projected_contributors(
    int count,
    const float* positions,
    const flashgs_adapter::cov3d_t* covariances,
    const float* opacities,
    const int64_t* semantic_ids,
    const float* view,
    const float* intrinsic,
    int width,
    int height,
    float near_plane,
    float far_plane,
    float support_sigma,
    float covariance_epsilon,
    int target_pixel_x,
    int target_pixel_y,
    int64_t expected_semantic_id,
    bool require_old_predicate_rejection,
    float* scores,
    cudaStream_t stream) {
  constexpr int threads = 256;
  score_contributors_kernel<<<
      (count + threads - 1) / threads,
      threads,
      0,
      stream>>>(
      count,
      positions,
      covariances,
      opacities,
      semantic_ids,
      view,
      intrinsic,
      width,
      height,
      near_plane,
      far_plane,
      support_sigma,
      covariance_epsilon,
      target_pixel_x,
      target_pixel_y,
      expected_semantic_id,
      require_old_predicate_rejection,
      scores);
}

void trace_candidates(
    int count,
    const float* positions,
    const flashgs_adapter::cov3d_t* covariances,
    const float* opacities,
    const float* view,
    const float* intrinsic,
    int width,
    int height,
    float near_plane,
    float far_plane,
    float support_sigma,
    float covariance_epsilon,
    int target_pixel_x,
    int target_pixel_y,
    const uint64_t* keys_unsorted,
    const uint32_t* values_unsorted,
    const uint64_t* keys_sorted,
    const uint32_t* values_sorted,
    const int2* ranges,
    const float2* points_xy,
    const float4* conic_opacity,
    int capacity,
    const int64_t* candidate_ids,
    int candidate_count,
    float* float_trace,
    int64_t* int_trace,
    cudaStream_t stream) {
  constexpr int threads = 64;
  trace_kernel<<<
      (candidate_count + threads - 1) / threads,
      threads,
      0,
      stream>>>(
      count,
      positions,
      covariances,
      opacities,
      view,
      intrinsic,
      width,
      height,
      near_plane,
      far_plane,
      support_sigma,
      covariance_epsilon,
      target_pixel_x,
      target_pixel_y,
      keys_unsorted,
      values_unsorted,
      keys_sorted,
      values_sorted,
      ranges,
      points_xy,
      conic_opacity,
      capacity,
      candidate_ids,
      candidate_count,
      float_trace,
      int_trace);
}

}  // namespace flashgs_debug
