#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>

struct GPUPowerData {
    std::string gpu_name;
    std::string uuid;
    double card_power;
    double tile0_power;
    double tile1_power;
};

class GPUPowerMonitor {
private:
    struct PowerDomainData {
        zes_pwr_handle_t handle;
        bool isCardLevel;
        int subdeviceId;
        zes_power_energy_counter_t lastCounter;
    };

    struct DeviceData {
        zes_device_handle_t device;
        std::string name;
        std::string uuid;
        std::vector<PowerDomainData> powerDomains;
    };

    std::vector<DeviceData> devices;
    bool initialized;

    void printError(const char* funcName, ze_result_t result) {
        std::cerr << "Error in " << funcName << ": " << result << std::endl;
    }

    std::string getDeviceUUID(zes_device_handle_t device) {
        zes_device_properties_t props = {};
        props.stype = ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        
        ze_result_t result = zesDeviceGetProperties(device, &props);
        if (result != ZE_RESULT_SUCCESS) {
            return "unknown";
        }
        
        std::string uuid;
        for (int i = 0; i < ZE_MAX_DEVICE_UUID_SIZE; i++) {
            char hex[3];
            sprintf(hex, "%02x", props.core.uuid.id[i]);
            uuid += hex;
        }
        return uuid;
    }

    bool initializeDevices() {
        ze_result_t result;

        // 1) Always initialize core driver, which also initializes Sysman
        //    when ZES_ENABLE_SYSMAN=1 is set in the environment.
        result = zeInit(0);
        if (result != ZE_RESULT_SUCCESS) {
            printError("zeInit", result);
            return false;
        }

        // 2) Initialize Sysman explicitly
        result = zesInit(0);
        if (result != ZE_RESULT_SUCCESS) {
            printError("zesInit", result);
            return false;
        }

        // 3) Discover Sysman drivers and devices
        uint32_t driverCount = 0;
        result = zesDriverGet(&driverCount, nullptr);
        if (result != ZE_RESULT_SUCCESS || driverCount == 0) {
            printError("zesDriverGet", result);
            return false;
        }

        std::vector<zes_driver_handle_t> drivers(driverCount);
        result = zesDriverGet(&driverCount, drivers.data());
        if (result != ZE_RESULT_SUCCESS) {
            printError("zesDriverGet", result);
            return false;
        }

        // 4) Enumerate devices for each driver
        for (uint32_t i = 0; i < driverCount; i++) {
            uint32_t deviceCount = 0;
            result = zesDeviceGet(drivers[i], &deviceCount, nullptr);
            if (result != ZE_RESULT_SUCCESS || deviceCount == 0) {
                continue;
            }

            std::vector<zes_device_handle_t> driverDevices(deviceCount);
            result = zesDeviceGet(drivers[i], &deviceCount, driverDevices.data());
            if (result != ZE_RESULT_SUCCESS) {
                continue;
            }

            // 5) Process each device
            for (uint32_t j = 0; j < deviceCount; j++) {
                DeviceData deviceData;
                deviceData.device = driverDevices[j];

                // Get device properties
                zes_device_properties_t props = {};
                props.stype = ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES;
                result = zesDeviceGetProperties(deviceData.device, &props);
                if (result == ZE_RESULT_SUCCESS) {
                    deviceData.name = std::string(props.core.name);
                } else {
                    deviceData.name = "Unknown Device";
                }

                deviceData.uuid = getDeviceUUID(deviceData.device);

                // 6) Enumerate power domains for this device
                uint32_t powerDomainCount = 0;
                result = zesDeviceEnumPowerDomains(deviceData.device, &powerDomainCount, nullptr);
                if (result != ZE_RESULT_SUCCESS || powerDomainCount == 0) {
                    std::cout << "No power domains found for device: " << deviceData.name << std::endl;
                    continue;
                }

                std::vector<zes_pwr_handle_t> powerHandles(powerDomainCount);
                result = zesDeviceEnumPowerDomains(deviceData.device, &powerDomainCount, powerHandles.data());
                if (result != ZE_RESULT_SUCCESS) {
                    continue;
                }

                // 7) Setup power domains
                for (uint32_t k = 0; k < powerDomainCount; k++) {
                    zes_power_properties_t powerProps = {};
                    powerProps.stype = ZES_STRUCTURE_TYPE_POWER_PROPERTIES;
                    
                    result = zesPowerGetProperties(powerHandles[k], &powerProps);
                    if (result != ZE_RESULT_SUCCESS) {
                        continue;
                    }

                    PowerDomainData powerDomain;
                    powerDomain.handle = powerHandles[k];
                    powerDomain.isCardLevel = powerProps.onSubdevice ? false : true;
                    powerDomain.subdeviceId = powerProps.onSubdevice ? powerProps.subdeviceId : -1;

                    // Initialize the energy counter
                    result = zesPowerGetEnergyCounter(powerDomain.handle, &powerDomain.lastCounter);
                    if (result == ZE_RESULT_SUCCESS) {
                        deviceData.powerDomains.push_back(powerDomain);
                        std::cout << "Added power domain - Card level: " << powerDomain.isCardLevel 
                                  << ", Subdevice ID: " << powerDomain.subdeviceId << std::endl;
                    }
                }

                if (!deviceData.powerDomains.empty()) {
                    devices.push_back(deviceData);
                    std::cout << "Added device: " << deviceData.name << " with " 
                              << deviceData.powerDomains.size() << " power domains" << std::endl;
                }
            }
        }

        return !devices.empty();
    }

public:
    GPUPowerMonitor() : initialized(false) {}

    bool initialize() {
        if (!initialized) {
            initialized = initializeDevices();
        }
        return initialized;
    }

    std::vector<GPUPowerData> getPowerReadings() {
        std::vector<GPUPowerData> readings;
        if (!initialized) return readings;

        for (auto& device : devices) {
            GPUPowerData data;
            data.gpu_name = device.name;
            data.uuid = device.uuid;
            data.card_power = -1;
            data.tile0_power = -1;
            data.tile1_power = -1;

            for (auto& domain : device.powerDomains) {
                zes_power_energy_counter_t currentCounter;
                ze_result_t result = zesPowerGetEnergyCounter(domain.handle, &currentCounter);
                if (result == ZE_RESULT_SUCCESS) {
                    // Handle wraparound (though this is simplified - proper wraparound handling is more complex)
                    uint64_t deltaTime = 0;
                    uint64_t deltaEnergy = 0;
                    
                    if (currentCounter.timestamp >= domain.lastCounter.timestamp) {
                        deltaTime = currentCounter.timestamp - domain.lastCounter.timestamp;
                    }
                    
                    if (currentCounter.energy >= domain.lastCounter.energy) {
                        deltaEnergy = currentCounter.energy - domain.lastCounter.energy;
                    }

                    if (deltaTime > 0) {
                        // Power in watts (energy is in microjoules, timestamp in microseconds)
                        double power = static_cast<double>(deltaEnergy) / static_cast<double>(deltaTime);
                        
                        if (domain.isCardLevel) {
                            data.card_power = power;
                        } else if (domain.subdeviceId == 0) {
                            data.tile0_power = power;
                        } else if (domain.subdeviceId == 1) {
                            data.tile1_power = power;
                        }
                    }

                    domain.lastCounter = currentCounter;
                }
            }
            readings.push_back(data);
        }

        return readings;
    }

    void printDeviceInfo() {
        if (!initialized) {
            std::cout << "Monitor not initialized" << std::endl;
            return;
        }

        std::cout << "Found " << devices.size() << " GPU devices:" << std::endl;
        for (const auto& device : devices) {
            std::cout << "  Device: " << device.name << std::endl;
            std::cout << "  UUID: " << device.uuid << std::endl;
            std::cout << "  Power domains: " << device.powerDomains.size() << std::endl;
        }
    }
};