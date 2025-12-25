"use client";

import { useState, useEffect } from "react";
import Sidebar from "../components/layout/Sidebar";
import MapComponent from "../components/layout/MapComponent";
import { fetchLocations } from "../services/api";
import type { LocationData, RadioOption } from "../types/location";

export default function MapPage() {
  const [searchText, setSearchText] = useState<string>("jhapa");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [locations, setLocations] = useState<LocationData[]>([]);
  const [selectedLocation, setSelectedLocation] = useState<LocationData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const radioOptions: RadioOption[] = [
    { value: "all", label: "All Categories" },
    { value: "park", label: "Parks" },
    { value: "landmark", label: "Landmarks" },
    { value: "museum", label: "Museums" },
    { value: "education", label: "Education" },
  ];

  const handleSearch = async () => {
    setLoading(true);
    try {
      const results = await fetchLocations(searchText, selectedCategory);
      const rawData = results.data;

      console.log("Raw API Data:", rawData);

      // FIX START: Handle both Array and Single Object responses
      let dataToProcess: any[] = [];

      if (Array.isArray(rawData)) {
        // Case 1: API returned a list of locations
        dataToProcess = rawData;
      } else if (rawData && typeof rawData === "object") {
        // Case 2: API returned a single location object (The issue in your screenshot)
        // We wrap it in an array so the map can render it
        dataToProcess = [rawData];
      }

      // Sanitize the data (ensure lat/lng are numbers)
      const sanitizedData = dataToProcess.map((loc: any) => ({
        ...loc,
        latitude: Number(loc.latitude),
        longitude: Number(loc.longitude),
      }));
      // FIX END

      console.log("Processed Locations:", sanitizedData);
      setLocations(sanitizedData);
      setSelectedLocation(null);
    } catch (error) {
      console.error("Error fetching locations:", error);
      setLocations([]);
    } finally {
      setLoading(false);
    }
  };

  const handleCategoryChange = (category: string) => {
    setSelectedCategory(category);
  };

  const handleMarkerClick = (location: LocationData) => {
    setSelectedLocation(location);
  };

  useEffect(() => {
    handleSearch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCategory]);

  return (
    <div className="flex h-screen w-screen bg-gray-100">
      <div className="w-96 h-full flex-shrink-0 relative z-20">
        <Sidebar
          searchValue={searchText}
          onSearchChange={setSearchText}
          onSearchSubmit={handleSearch}
          radioOptions={radioOptions}
          selectedRadio={selectedCategory}
          onRadioChange={handleCategoryChange}
          selectedLocation={selectedLocation}
        />
      </div>

      <div className="flex-1 h-full p-4 relative z-10">
        {loading ? (
          <div className="w-full h-full flex items-center justify-center bg-white rounded-lg shadow-lg">
            <div className="text-center">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
              <p className="text-gray-600">Loading locations...</p>
            </div>
          </div>
        ) : (
          <MapComponent
            locations={locations}
            onMarkerClick={handleMarkerClick}
            center={
              locations.length > 0
                ? [locations[0].latitude, locations[0].longitude]
                : [27.6253, 85.5561]
            }
            zoom={12}
          />
        )}
      </div>
    </div>
  );
}
