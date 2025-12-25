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

    </div>
  );
}
