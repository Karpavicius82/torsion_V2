cmake_minimum_required(VERSION 3.24)
project(antigravity_core VERSION 0.26.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Drogon CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)

function(agx_apply_target_defaults target_name)
    target_compile_features(${target_name} PUBLIC cxx_std_20)
    target_include_directories(${target_name} PRIVATE src)

    if(MSVC)
        target_compile_options(${target_name} PRIVATE
            /W4 /permissive- /EHsc
            /Zc:__cplusplus
            /Zc:preprocessor
            /utf-8
            /MP
            /wd4251 /wd4275 /wd4324
        )
        target_compile_definitions(${target_name} PRIVATE
            _CRT_SECURE_NO_WARNINGS
            NOMINMAX
            WIN32_LEAN_AND_MEAN
        )
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # if(WIN32)
    #     add_custom_command(TARGET ${target_name} POST_BUILD
    #         COMMAND ${CMAKE_COMMAND} -E copy_if_different
    #             $<TARGET_RUNTIME_DLLS:${target_name}>
    #             $<TARGET_FILE_DIR:${target_name}>
    #         COMMAND_EXPAND_LISTS
    #         COMMENT "Copying runtime DLLs for ${target_name}..."
    #     )
    # endif()
endfunction()

add_executable(antigravity_core
    src/app/main.cpp
    src/app/bootstrap.cpp
    src/application/service.cpp
    src/application/service_batch.cpp
    src/application/service_runs.cpp
    src/application/service_supervisor.cpp
    src/application/argument_resolver.cpp
    src/application/office_bridge_client.cpp
    src/application/response_composer.cpp
    src/bus/envelope.cpp
    src/bus/channel_registry.cpp
    src/bus/channel_router.cpp
    src/bus/subscription_store.cpp
    src/bus/stream_manager.cpp
    src/bus/qos_policy.cpp
    src/control/control_kernel.cpp
    src/control/living_silicon.cpp
    src/control/mutation_arbiter.cpp
    src/control/state_encoder.cpp
    src/control/step_queue.cpp
    src/control/lease_manager.cpp
    src/control/run_registry.cpp
    src/control/recovery_coordinator.cpp
    src/control/mode_policy.cpp
    src/control_plane/types.cpp
    src/control_plane/service.cpp
    src/control_plane/worker_spawner.cpp
    src/control_plane/inline_worker_pool.cpp
    src/domain/message_classifier.cpp
    src/domain/orchestrator.cpp
    src/domain/reducer.cpp
    src/execution/executor_registry.cpp
    src/execution/inline_executor.cpp
    src/execution/openai_client.cpp
    src/execution/worker_host.cpp
    src/execution/worker_pool.cpp
    src/execution/browser_control.cpp
    src/infrastructure/config/runtime_paths.cpp
    src/infrastructure/config/runtime_json.cpp
    src/infrastructure/config/runtime_settings.cpp
    src/infrastructure/storage/sqlite_store.cpp
    src/transport/http_server.cpp
    src/transport/mcp_router.cpp
    src/security/pin_manager.cpp
    src/security/oauth2_server.cpp
    src/system/hygiene.cpp
    src/system/watchdog.cpp
    src/bridge/bridge_controller.cpp
    src/util/hash.cpp
    src/util/id.cpp
    src/util/process_executor.cpp
    src/util/time.cpp
)

target_link_libraries(antigravity_core PRIVATE
    Drogon::Drogon
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    unofficial::sqlite3::sqlite3
)
agx_apply_target_defaults(antigravity_core)

# ---- WINDOWS SERVICE (AGXService) ----
if(WIN32)
    add_executable(AGXService
        src/service/AGXService.cpp
        src/service/child_process.cpp
        src/service/health_probe.cpp
        src/infrastructure/config/runtime_paths.cpp
        src/infrastructure/config/runtime_json.cpp
        src/infrastructure/config/runtime_settings.cpp
    )
    target_link_libraries(AGXService PRIVATE
        nlohmann_json::nlohmann_json
        winhttp
        advapi32
    )
    agx_apply_target_defaults(AGXService)

    # Windows BCrypt (SHA-256 be OpenSSL, reikalingas pin_manager)
    target_link_libraries(antigravity_core PRIVATE bcrypt winhttp)
endif()

# ---- NATIVE WORKER RUNTIME ----
if(WIN32)
    add_executable(agx_worker_runtime
        src/worker_runtime/main.cpp
        src/worker_runtime/mcp_client.cpp
        src/worker_runtime/worker_loop.cpp
        src/worker_runtime/role_handlers.cpp
    )
    target_link_libraries(agx_worker_runtime PRIVATE
        nlohmann_json::nlohmann_json
        spdlog::spdlog
        winhttp
    )
    agx_apply_target_defaults(agx_worker_runtime)
endif()

# ---- NATIVE SUPERVISOR ----
if(WIN32)
    add_executable(agx_supervisor
        src/supervisor/main.cpp
    )
    target_link_libraries(agx_supervisor PRIVATE
        spdlog::spdlog
    )
    agx_apply_target_defaults(agx_supervisor)
endif()

option(ANTIGRAVITY_BUILD_OFFICE_BRIDGE "Build office_bridge_v3 together with core" ON)

if(ANTIGRAVITY_BUILD_OFFICE_BRIDGE)
    add_subdirectory(office_bridge_v3)
endif()

# ---- BRIDGE SMOKE TEST ----
if(WIN32)
    add_executable(bridge_smoke_test tests/bridge_smoke_test.cpp)
    target_link_libraries(bridge_smoke_test PRIVATE ws2_32)
    set_target_properties(bridge_smoke_test PROPERTIES
        CXX_STANDARD 17
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
    )

    add_executable(full_test_suite tools/full_test_suite.cpp)
    target_link_libraries(full_test_suite PRIVATE winhttp)
    set_target_properties(full_test_suite PROPERTIES
        CXX_STANDARD 17
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
    )

    add_executable(office_bridge_test tools/office_bridge_test.cpp)
    target_link_libraries(office_bridge_test PRIVATE winhttp)
    set_target_properties(office_bridge_test PROPERTIES
        CXX_STANDARD 17
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
    )

    add_executable(agx_teismai_crawler tools/teismai_crawler.cpp)
    target_link_libraries(agx_teismai_crawler PRIVATE winhttp)
    set_target_properties(agx_teismai_crawler PROPERTIES
        CXX_STANDARD 17
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
    )

    # ---- REGRESSION TESTS (in-memory, deterministic, build-blocking) ----
    add_executable(antigravity_regression_tests
        tools/regression_tests.cpp
        src/domain/reducer.cpp
        src/util/id.cpp
        src/util/time.cpp
    )
    target_link_libraries(antigravity_regression_tests PRIVATE
        nlohmann_json::nlohmann_json
        spdlog::spdlog
        unofficial::sqlite3::sqlite3
    )
    agx_apply_target_defaults(antigravity_regression_tests)
endif()
