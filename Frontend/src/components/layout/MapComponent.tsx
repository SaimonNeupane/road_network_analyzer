"use client";

import { useState, useRef, useEffect } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  useMap,
  useMapEvents,
} from "react-leaflet";
import L from "leaflet";

interface LocationData {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  description: string;
  category: string;
}

interface MapComponentProps {
  locations: LocationData[];
  center: [number, number];
  zoom: number;
  onMarkerClick: (location: LocationData) => void;
}

// Custom component to handle map events and selected location
function MapController({
  selectedLocation,
  onZoomChange,
  onCenterChange,
}: {
  selectedLocation: LocationData | null;
  onZoomChange: (zoom: number) => void;
  onCenterChange: (center: [number, number]) => void;
}) {
  const map = useMap();

  useEffect(() => {
    if (selectedLocation) {
      map.flyTo([selectedLocation.latitude, selectedLocation.longitude], 15, {
        duration: 1.5,
      });
    }
  }, [selectedLocation, map]);

  useMapEvents({
    zoomend: () => {
      onZoomChange(map.getZoom());
    },
    moveend: () => {
      const center = map.getCenter();
      onCenterChange([center.lat, center.lng]);
    },
  });

  return null;
}

// Custom marker icons based on category and selection
function getMarkerIcon(category: string, isSelected: boolean) {
  const colors: Record<string, string> = {
    restaurant: "#ef4444",
    hotel: "#3b82f6",
    attraction: "#10b981",
    shopping: "#f59e0b",
    default: "#8b5cf6",
  };

  const color = colors[category.toLowerCase()] || colors.default;
  const scale = isSelected ? 1.3 : 1;

  return L.divIcon({
    className: "custom-marker",
    html: `
        <div style="transform: scale(${scale}); transition: transform 0.2s;">
          <svg width="32" height="42" viewBox="0 0 32 42" style="filter: drop-shadow(0 4px 6px rgba(0,0,0,0.3));">
            <path d="M16 0C7.163 0 0 7.163 0 16c0 8.837 16 26 16 26s16-17.163 16-26C32 7.163 24.837 0 16 0z" 
                  fill="${color}" />
            <circle cx="16" cy="16" r="6" fill="white" />
          </svg>
        </div>
      `,
    iconSize: [32, 42],
    iconAnchor: [16, 42],
    popupAnchor: [0, -42],
  });
}

export default function MapComponent({
  locations,
  center,
  zoom,
  onMarkerClick,
}: MapComponentProps) {
  const [selectedLocation, setSelectedLocation] = useState<LocationData | null>(
    null
  );
  const [currentZoom, setCurrentZoom] = useState(zoom);
  const [currentCenter, setCurrentCenter] = useState(center);
  const [mapStyle, setMapStyle] = useState("streets");
  const [showControls, setShowControls] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const mapRef = useRef<L.Map | null>(null);

  // Filter locations based on search
  const filteredLocations =
    searchTerm.trim() === ""
      ? locations
      : locations.filter(
          (loc) =>
            loc.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            loc.category.toLowerCase().includes(searchTerm.toLowerCase()) ||
            loc.description.toLowerCase().includes(searchTerm.toLowerCase())
        );
  const handleMarkerClick = (location: LocationData) => {
    setSelectedLocation(location);
    onMarkerClick(location);
  };

  const tileLayerUrl = (() => {
    const styles: Record<string, string> = {
      streets: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      satellite:
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      dark: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      light: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    };
    return styles[mapStyle] || styles.streets;
  })();

  const categoryColors: Record<string, string> = {
    restaurant: "bg-red-500",
    hotel: "bg-blue-500",
    attraction: "bg-green-500",
    shopping: "bg-orange-500",
  };

  const getCategoryColor = (category: string) => {
    return categoryColors[category.toLowerCase()] || "bg-purple-500";
  };

  return (
    <div className="relative w-full h-full">
      {/* Search Bar */}
      <div className="absolute top-4 left-4 z-1000 w-80">
        <input
          type="text"
          placeholder="Search locations..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full px-4 py-2 rounded-lg shadow-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        {searchTerm && (
          <div className="mt-2 bg-white rounded-lg shadow-lg max-h-60 overflow-y-auto">
            {filteredLocations.map((loc) => (
              <div
                key={loc.id}
                onClick={() => {
                  handleMarkerClick(loc);
                  setSearchTerm("");
                }}
                className="p-3 hover:bg-gray-100 cursor-pointer border-b border-gray-200 last:border-0"
              >
                <div className="font-semibold text-sm">{loc.name}</div>
                <div className="text-xs text-gray-600">{loc.category}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Map Style Selector */}
      {showControls && (
        <div className="absolute top-4 right-4 z-[1000] bg-white rounded-lg shadow-lg p-2">
          <div className="flex gap-2">
            {["streets", "satellite", "dark", "light"].map((style) => (
              <button
                key={style}
                onClick={() => setMapStyle(style)}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                  mapStyle === style
                    ? "bg-blue-500 text-white"
                    : "bg-gray-100 hover:bg-gray-200"
                }`}
              >
                {style.charAt(0).toUpperCase() + style.slice(1)}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Info Panel */}
      {selectedLocation && (
        <div className="absolute bottom-4 left-4 z-[1000] w-80 bg-white rounded-lg shadow-xl p-4">
          <button
            onClick={() => setSelectedLocation(null)}
            className="absolute top-2 right-2 text-gray-400 hover:text-gray-600"
          >
            ✕
          </button>
          <h3 className="font-bold text-lg mb-2">{selectedLocation.name}</h3>
          <p className="text-sm text-gray-600 mb-3">
            {selectedLocation.description}
          </p>
          <div className="flex items-center gap-2 mb-3">
            <span
              className={`text-xs text-white px-3 py-1 rounded-full ${getCategoryColor(
                selectedLocation.category
              )}`}
            >
              {selectedLocation.category}
            </span>
          </div>
          <div className="text-xs text-gray-500">
            📍 {selectedLocation.latitude.toFixed(4)},{" "}
            {selectedLocation.longitude.toFixed(4)}
          </div>
        </div>
      )}

      {/* Stats Bar */}
      {showControls && (
        <div className="absolute bottom-4 right-4 z-[1000] bg-white rounded-lg shadow-lg px-4 py-2 flex items-center gap-4 text-xs">
          <div>
            <span className="text-gray-500">Zoom:</span>
            <span className="font-mono ml-1 font-semibold">
              {currentZoom.toFixed(1)}
            </span>
          </div>
          <div className="border-l border-gray-300 pl-4">
            <span className="text-gray-500">Locations:</span>
            <span className="font-mono ml-1 font-semibold">
              {filteredLocations.length}
            </span>
          </div>
          <button
            onClick={() => setShowControls(false)}
            className="ml-2 text-gray-400 hover:text-gray-600"
          >
            ⚙️
          </button>
        </div>
      )}

      {/* Toggle Controls Button (when hidden) */}
      {!showControls && (
        <button
          onClick={() => setShowControls(true)}
          className="absolute bottom-4 right-4 z-[1000] bg-white rounded-lg shadow-lg p-3 hover:bg-gray-50"
        >
          ⚙️
        </button>
      )}

      {/* Legend */}
      <div className="absolute top-20 right-4 z-[1000] bg-white rounded-lg shadow-lg p-3 w-40">
        <h4 className="font-semibold text-xs mb-2">Categories</h4>
        {Array.from(new Set(locations.map((l) => l.category))).map(
          (category) => (
            <div key={category} className="flex items-center gap-2 mb-1">
              <div
                className={`w-3 h-3 rounded-full ${getCategoryColor(category)}`}
              />
              <span className="text-xs">{category}</span>
            </div>
          )
        )}
      </div>

      {/* Map Container */}
      <MapContainer
        center={currentCenter}
        zoom={zoom}
        className="w-full h-full"
        ref={mapRef}
        zoomControl={false}
      >
        <TileLayer url={tileLayerUrl} attribution="© Map Data" />

        <MapController
          selectedLocation={selectedLocation}
          onZoomChange={setCurrentZoom}
          onCenterChange={setCurrentCenter}
        />

        {filteredLocations.map((location) => (
          <Marker
            key={location.id}
            position={[location.latitude, location.longitude]}
            icon={getMarkerIcon(
              location.category,
              selectedLocation?.id === location.id
            )}
            eventHandlers={{
              click: () => handleMarkerClick(location),
            }}
          >
            <Popup>
              <div className="p-2">
                <h3 className="font-bold text-sm mb-1">{location.name}</h3>
                <p className="text-xs text-gray-600 mb-2">
                  {location.description}
                </p>
                <span
                  className={`text-xs text-white px-2 py-1 rounded ${getCategoryColor(
                    location.category
                  )}`}
                >
                  {location.category}
                </span>
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
}
