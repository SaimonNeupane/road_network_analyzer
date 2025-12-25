"use client";
import { useState } from "react";
import Sidebar from "../components/layout/Sidebar";
import MapComponent from "../components/layout/MapComponent";
import { fetchLocations, fetchProposedRoads, fetchRoads } from "../services/api";
import type { LocationData, RadioOption } from "../types/location";

export default function MapPage() {
  const [searchText, setSearchText] = useState<string>("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [locations, setLocations] = useState<LocationData[]>([]);

  // Input State
  const [district, setDistrict] = useState("Kavre");
  const [place, setPlace] = useState("Dhulikhel");

  // Map Data State
  const [existingRoads, setExistingRoads] = useState<any>(null); // Layer 1: Grey roads
  const [proposedRoads, setProposedRoads] = useState<any>(null); // Layer 2: Colored proposals

  const [selectedLocation, setSelectedLocation] = useState<LocationData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const radioOptions: RadioOption[] = [
    { value: "all", label: "All Categories" },
    { value: "park", label: "Parks" },
    { value: "landmark", label: "Landmarks" },
    { value: "education", label: "Education" },
  ];

  const handleSearch = async () => {
    if (!district || !place) {
      alert("Please enter both District and Place name");
      return;
    }

    setLoading(true);
    setExistingRoads(null);
    setProposedRoads(null);
    setLocations([]);

    try {
      // 1. Fetch Locations (for markers)
      const locationPromise = fetchLocations(district, place);

      // 2. Fetch Existing Roads (standard OSM data)
      const roadsPromise = fetchRoads(place);

      // 3. Fetch Proposed Roads (Python script result)
      console.log(`Running analysis for: ${place}, ${district}...`);
      const proposedPromise = fetchProposedRoads(place, district);

      // Run all requests in parallel
      const [locationRes, roadsRes, proposedRes] = await Promise.all([
        locationPromise,
        roadsPromise,
        proposedPromise
      ]);

      // --- Process Markers ---
      const rawData = locationRes.data;
      const dataToProcess = Array.isArray(rawData) ? rawData : [rawData];
      const sanitizedData = dataToProcess.map((loc: any) => ({
        ...loc,
        latitude: Number(loc.latitude),
        longitude: Number(loc.longitude),
      }));
      setLocations(sanitizedData);

      // --- Process Existing Roads ---
      if (roadsRes.data) {
        setExistingRoads(roadsRes.data);
      }

      // --- Process Proposed Roads ---
      if (proposedRes.data) {
        console.log("Proposed Data Received:", proposedRes.data);
        setProposedRoads(proposedRes.data);
      }

      setSelectedLocation(null);

    } catch (error) {
      console.error("Error fetching data:", error);
      alert("An error occurred while fetching map data. Check console for details.");
    } finally {
      setLoading(false);
    }
  };

  const handleCategoryChange = (category: string) => {
    setSelectedCategory(category);
  };

  return (
    <div className="flex h-screen w-screen bg-gray-100">
      {/* Sidebar Area */}
      <div className="w-96 h-full flex-shrink-0 relative z-20 flex flex-col bg-white shadow-lg p-4">
        <h2 className="text-xl font-bold mb-4">Road Planner</h2>

        {/* INPUTS */}
        <div className="flex flex-col gap-3 mb-6">
          <div>
            <label className="text-xs font-bold text-gray-500 uppercase">District</label>
            <input
              className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 outline-none"
              value={district}
              onChange={(e) => setDistrict(e.target.value)}
              placeholder="e.g. Kavre"
            />
          </div>
          <div>
            <label className="text-xs font-bold text-gray-500 uppercase">Place Name</label>
            <input
              className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 outline-none"
              value={place}
              onChange={(e) => setPlace(e.target.value)}
              placeholder="e.g. Dhulikhel"
            />
          </div>
          <button
            onClick={handleSearch}
            disabled={loading}
            className={`w-full py-2 rounded text-white transition ${loading ? "bg-gray-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"
              }`}
          >
            {loading ? "Calculating..." : "Find Proposed Roads"}
          </button>
        </div>

        <div className="flex-1 overflow-auto">
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
      </div>

      {/* Map Area */}
      <div className="flex-1 h-full p-4 relative z-10">
        {loading ? (
          <div className="w-full h-full flex items-center justify-center bg-white rounded-lg shadow-lg">
            <div className="text-center">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
              <p className="text-gray-600 font-medium">Running GNN Inference...</p>
              <p className="text-xs text-gray-400">This may take a few seconds</p>
            </div>
          </div>
        ) : (
          <MapComponent
            locations={locations}
            existingRoads={existingRoads} // Pass existing layer
            proposedRoads={proposedRoads} // Pass proposed layer
            onMarkerClick={setSelectedLocation}
            center={
              locations.length > 0
                ? [locations[0].latitude, locations[0].longitude]
                : [27.6253, 85.5561]
            }
            zoom={13}
          />
        )}
      </div>
    </div>
  );
}
