"use client";
import { useState, useEffect } from "react";
import Sidebar from "../components/layout/Sidebar";
import MapComponent from "../components/layout/MapComponent";
import { fetchLocations, fetchRoads } from "../services/api"; // Import fetchRoads
import type { LocationData, RadioOption } from "../types/location";

export default function MapPage() {
  const [searchText, setSearchText] = useState<string>("jhapa");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [locations, setLocations] = useState<LocationData[]>([]);
  const [roadData, setRoadData] = useState<any>(null); // <--- New State for Roads
  const [selectedLocation, setSelectedLocation] = useState<LocationData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // ... [Keep Radio Options same] ...
  const radioOptions: RadioOption[] = [
    { value: "all", label: "All Categories" },
    { value: "park", label: "Parks" },
    { value: "landmark", label: "Landmarks" },
    { value: "museum", label: "Museums" },
    { value: "education", label: "Education" },
  ];

  const handleSearch = async () => {
    setLoading(true);
    setRoadData(null); // Reset roads while loading new ones

    try {
      // 1. Fetch Locations
      // We do this separately or concurrently. 
      // Let's do it concurrently to save time, but wrap roads in a try-catch 
      // so if roads fail (e.g. 404), markers still load.

      const locationPromise = fetchLocations(searchText, selectedCategory);
      const roadPromise = fetchRoads(searchText).catch(err => {
        console.warn("Could not fetch roads:", err);
        return { data: null }; // Return null if roads fail
      });

      const [locationRes, roadRes] = await Promise.all([locationPromise, roadPromise]);

      // --- Process Location Data ---
      const rawData = locationRes.data;
      let dataToProcess: any[] = [];
      if (Array.isArray(rawData)) {
        dataToProcess = rawData;
      } else if (rawData && typeof rawData === "object") {
        dataToProcess = [rawData];
      }
      const sanitizedData = dataToProcess.map((loc: any) => ({
        ...loc,
        latitude: Number(loc.latitude),
        longitude: Number(loc.longitude),
      }));
      setLocations(sanitizedData);

      // --- Process Road Data ---
      if (roadRes && roadRes.data) {
        setRoadData(roadRes.data);
      }

      setSelectedLocation(null);
    } catch (error) {
      console.error("Error fetching data:", error);
      setLocations([]);
    } finally {
      setLoading(false);
    }
  };

  // ... [Keep remaining handlers same] ...
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
              <p className="text-gray-600">Loading locations & roads...</p>
            </div>
          </div>
        ) : (
          <MapComponent
            locations={locations}
            roadData={roadData} // <--- Pass the road data here
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
