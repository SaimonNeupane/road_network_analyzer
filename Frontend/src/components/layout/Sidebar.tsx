"use client";
import SearchBox from "../control/SearchBox";
import RadioFilter from "../control/RadioFilter";
import LocationDescription from "../info/LocationDescription";
import type { LocationData, RadioOption } from "../../types/location";

interface SidebarProps {
  searchValue: string;
  onSearchChange: (value: string) => void;
  onSearchSubmit: () => void;
  radioOptions: RadioOption[];
  selectedRadio: string;
  onRadioChange: (value: string) => void;
  selectedLocation: LocationData | null;
}

export default function Sidebar({
  searchValue,
  onSearchChange,
  onSearchSubmit,
  radioOptions,
  selectedRadio,
  onRadioChange,
  selectedLocation,
}: SidebarProps) {
  return (
    <div className="w-full h-full bg-white p-6 overflow-y-auto shadow-lg">
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Map Explorer
          </h2>
          <SearchBox
            value={searchValue}
            onChange={onSearchChange}
            onSubmit={onSearchSubmit}
          />
        </div>

        <div className="border-t border-gray-200 pt-6">
          <RadioFilter
            options={radioOptions}
            selectedValue={selectedRadio}
            onChange={onRadioChange}
          />
        </div>

        <div className="border-t border-gray-200 pt-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            Location Details
          </h3>
          <LocationDescription location={selectedLocation} />
        </div>
      </div>
    </div>
  );
}
