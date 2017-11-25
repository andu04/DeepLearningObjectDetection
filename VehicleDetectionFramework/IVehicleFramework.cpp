#include "VehicleFramework.h"

std::shared_ptr<IVehicleFramework> IVehicleFramework::Produce()
{
	return std::shared_ptr<IVehicleFramework>(new VehicleFramework());
}

