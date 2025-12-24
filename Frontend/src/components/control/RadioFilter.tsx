"use client";
import type { RadioOption } from "../../types/location";

interface RadioFilterProps {
  options: RadioOption[];
  selectedValue: string;
  onChange: (value: string) => void;
  label?: string;
}

export default function RadioFilter({
  options,
  selectedValue,
  onChange,
  label = "Filter by Category",
}: RadioFilterProps) {
  return (
    <div className="w-full">
      <label className="block text-sm font-semibold text-gray-700 mb-3">
        {label}
      </label>
      <div className="space-y-2">
        {options.map((option) => (
          <label
            key={option.value}
            className="flex items-center space-x-3 cursor-pointer group"
          >
            <input
              type="radio"
              value={option.value}
              checked={selectedValue === option.value}
              onChange={(e) => onChange(e.target.value)}
              className="w-4 h-4 text-blue-600 focus:ring-2 focus:ring-blue-500 cursor-pointer"
            />
            <span className="text-gray-700 group-hover:text-gray-900 transition-colors">
              {option.label}
            </span>
          </label>
        ))}
      </div>
    </div>
  );
}
