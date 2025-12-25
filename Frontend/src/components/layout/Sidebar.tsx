"use client";

import LocationDescription from "../info/LocationDescription";
import RadioFilter from "../control/RadioFilter";
import type { LocationData, RadioOption } from "../../types/location";

interface SidebarProps {
  searchName: string;
  searchCategory: string;
  onSearchNameChange: (value: string) => void;
  onSearchCategoryChange: (value: string) => void;
  onSearchSubmit: () => void;
  radioOptions: RadioOption[];
  selectedRadio: string;
  onRadioChange: (value: string) => void;
  selectedLocation: LocationData | null;
}

export default function Sidebar({
  searchName,
  searchCategory,
  onSearchNameChange,
  onSearchCategoryChange,
  onSearchSubmit,
  radioOptions,
  selectedRadio,
  onRadioChange,
  selectedLocation,
}: SidebarProps) {
  return (
    <div className="w-full h-full bg-white p-6 overflow-y-auto shadow-lg">
      <div className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-900">Map Explorer</h2>

        {/* Search Fields */}
        <div className="space-y-3">
          <input
            type="text"
            value={searchName}
            onChange={(e) => onSearchNameChange(e.target.value)}
            placeholder="Search by location name"
            className="w-full px-4 py-2 border rounded-lg"
          />

          <input
            type="text"
            value={searchCategory}
            onChange={(e) => onSearchCategoryChange(e.target.value)}
            placeholder="Search by category"
            className="w-full px-4 py-2 border rounded-lg"
          />

          {/* Single Search Button */}
          <button
            onClick={onSearchSubmit}
            className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700"
          >
            Search
          </button>
        </div>

        {/* Radio Filter */}
        <div className="border-t pt-6">
          <RadioFilter
            options={radioOptions}
            selectedValue={selectedRadio}
            onChange={onRadioChange}
          />
        </div>

        {/* Location Details */}
        <div className="border-t pt-6">
          <h3 className="text-lg font-semibold mb-3">Location Details</h3>
          <LocationDescription location={selectedLocation} />
        </div>
      </div>
    </div>
  );
}
