#pragma once

#ifndef VLLM_RCC_PROFILE
#define VLLM_RCC_PROFILE 0
#endif

// Profile 0 compiles every guarded probe out.
#define VLLM_RCC_PROFILE_ENABLED(profile_id) \
  (VLLM_RCC_PROFILE == (profile_id))
