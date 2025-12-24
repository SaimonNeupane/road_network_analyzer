import type { LocationData } from "../../types/location";

interface LocationDescriptionProps {
  location: LocationData | null;
}

export default function LocationDescription({
  location,
}: LocationDescriptionProps) {
  if (!location) {
    return (
      <div className="w-full p-4 bg-gray-50 rounded-lg border border-gray-200">
        <p className="text-gray-500 text-center">
          Click on a marker to view location details
        </p>
      </div>
    );
  }

  return (
    <div className="w-full p-4 bg-white rounded-lg border border-gray-200 shadow-sm">
      <h3 className="text-xl font-bold text-gray-900 mb-2">{location.name}</h3>
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-gray-600">Category:</span>
          <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full font-medium">
            {location.category}
          </span>
        </div>
        <p className="text-gray-700 leading-relaxed">{location.description}</p>
        <div className="pt-2 border-t border-gray-200 mt-3">
          <p className="text-xs text-gray-500">
            Coordinates: {location.latitude.toFixed(6)},{" "}
            {location.longitude.toFixed(6)}
          </p>
        </div>
      </div>
    </div>
  );
}
