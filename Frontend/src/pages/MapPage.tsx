"use client";

import { useState, useEffect } from "react";
import Sidebar from "../components/layout/Sidebar";
import MapComponent from "../components/layout/MapComponent";
import { fetchLocations } from "../services/api";
import type { LocationData, RadioOption } from "../types/location";

export default function MapPage() {
  const [searchText, setSearchText] = useState<string>("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [locations, setLocations] = useState<LocationData[]>([]);
  const [selectedLocation, setSelectedLocation] = useState<LocationData | null>(
    'dhulikhel'
  );
  const [loading, setLoading] = useState<boolean>(false);

  const radioOptions: RadioOption[] = [
    { value: "all", label: "All Categories" },
    { value: "park", label: "Parks" },
    { value: "landmark", label: "Landmarks" },
    { value: "museum", label: "Museums" },
  ];

  const handleSearch = async () => {
    setLoading(true);
    try {
      const results = await fetchLocations(searchText, selectedCategory);
      console.log(results)
      // setLocations(results);
      setSelectedLocation(null);
    } catch (error) {
      console.error("Error fetching locations:", error);
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
  }, [selectedCategory]);

  return (
    <div className="flex h-screen w-screen bg-gray-100">
      {/* Sidebar */}
      <div className="w-96 h-full flex-shrink-0">
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

      {/* Map Container */}
      <div className="flex-1 h-full p-4">
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
            center={[27.6253, 85.5561]}
            zoom={12}
          />
        )}
      </div>
    </div>
  );
}
