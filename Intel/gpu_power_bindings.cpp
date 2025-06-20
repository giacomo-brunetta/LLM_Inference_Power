#include "/usr/include/pybind11/pybind11.h"
#include "/usr/include/pybind11/stl.h"
#include "gpu_power.hpp"

namespace py = pybind11;

PYBIND11_MODULE(gpu_power, m) {
    py::class_<GPUPowerData>(m, "GPUPowerData")
        .def_readwrite("gpu_name", &GPUPowerData::gpu_name)
        .def_readwrite("uuid", &GPUPowerData::uuid)
        .def_readwrite("card_power", &GPUPowerData::card_power)
        .def_readwrite("tile0_power", &GPUPowerData::tile0_power)
        .def_readwrite("tile1_power", &GPUPowerData::tile1_power);

    py::class_<GPUPowerMonitor>(m, "GPUPowerMonitor")
        .def(py::init<>())
        .def("initialize", &GPUPowerMonitor::initialize)
        .def("get_power_readings", &GPUPowerMonitor::getPowerReadings);
}