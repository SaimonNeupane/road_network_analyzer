"use client";

import { useState, useEffect } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  useMap,
  useMapEvents,
  GeoJSON, // <--- 1. Import GeoJSON
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Interface definitions...
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
  roadData?: any; // <--- 2. Add roadData prop (Type 'any' for GeoJSON feature collection)
}

// ... [Keep MapController and getMarkerIcon exactly as they were] ...
function MapController({
  selectedLocation,
  center,
  onZoomChange,
  onCenterChange,
}: {
  selectedLocation: LocationData | null;
  center: [number, number];
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

  useEffect(() => {
    map.flyTo(center, map.getZoom());
  }, [center, map]);

  useMapEvents({
    zoomend: () => onZoomChange(map.getZoom()),
    moveend: () => {
      const c = map.getCenter();
      onCenterChange([c.lat, c.lng]);
    },
  });

  return null;
}

function getMarkerIcon(category: string, isSelected: boolean) {
  // ... [Keep your existing icon logic] ...
  const colors: Record<string, string> = {
    restaurant: "#ef4444",
    hotel: "#3b82f6",
    attraction: "#10b981",
    shopping: "#f59e0b",
    education: "#8b5cf6",
    park: "#22c55e",
    default: "#8b5cf6",
  };

  const color = colors[category.toLowerCase()] || colors.default;
  const scale = isSelected ? 1.3 : 1;

  return L.divIcon({
    className: "bg-transparent border-none",
    html: `
        <div style="transform: scale(${scale}); transition: transform 0.2s; display: flex; align-items: center; justify-content: center;">
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
  roadData, // <--- 3. Destructure roadData
}: MapComponentProps) {
  const [selectedLocation, setSelectedLocation] = useState<LocationData | null>(null);
  const [currentZoom, setCurrentZoom] = useState(zoom);
  const [currentCenter, setCurrentCenter] = useState(center);
  const [mapStyle, setMapStyle] = useState("streets");
  const [showControls, setShowControls] = useState(true);

  // Style for the roads (Blue lines)
  const roadStyle = {
    color: mapStyle == "streets" || mapStyle == "light" ? "#000000" : "#FF9800",

    // satellite: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    // // Tailwind blue-600
    weight: 1,
    opacity: 0.6,
  };

  useEffect(() => {
    setCurrentCenter(center);
  }, [center]);

  const handleMarkerClick = (location: LocationData) => {
    setSelectedLocation(location);
    onMarkerClick(location);
  };

  const tileLayerUrl = (() => {
    // ... [Keep your tile logic] ...
    const styles: Record<string, string> = {
      streets: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      satellite: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      dark: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      light: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    };
    return styles[mapStyle] || styles.streets;
  })();

  const getCategoryColor = (category: string) => {
    // ... [Keep your category color logic] ...
    const colors: Record<string, string> = {
      restaurant: "bg-red-500",
      hotel: "bg-blue-500",
      attraction: "bg-green-500",
      shopping: "bg-orange-500",
      education: "bg-purple-500",
      park: "bg-green-600",
    };
    return colors[category.toLowerCase()] || "bg-gray-500";
  };

  return (
    <div className="relative w-full h-full rounded-lg overflow-hidden shadow-xl border border-gray-200">
      {/* ... [Keep Controls and Info Panel logic unchanged] ... */}
      {showControls && (
        <div className="absolute top-4 right-4 z-[1000] bg-white rounded-lg shadow-lg p-2">
          <div className="flex gap-2">
            {["streets", "satellite", "dark", "light"].map((style) => (
              <button
                key={style}
                onClick={() => setMapStyle(style)}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${mapStyle === style
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

      {selectedLocation && (
        // ... [Keep Popup logic unchanged] ...
        <div className="absolute bottom-4 left-4 z-[1000] w-80 bg-white rounded-lg shadow-xl p-4 animate-in fade-in slide-in-from-bottom-4 duration-300">
          <button
            onClick={() => setSelectedLocation(null)}
            className="absolute top-2 right-2 text-gray-400 hover:text-gray-600"
          >
            ✕
          </button>
          <h3 className="font-bold text-lg mb-2 text-gray-800">{selectedLocation.name}</h3>
          <p className="text-sm text-gray-600 mb-3 leading-relaxed">
            {selectedLocation.description}
          </p>
          <div className="flex items-center gap-2 mb-3">
            <span className={`text-xs text-white px-3 py-1 rounded-full capitalize ${getCategoryColor(selectedLocation.category)}`}>
              {selectedLocation.category}
            </span>
          </div>
          <div className="text-xs text-gray-400 font-mono">
            📍 {Number(selectedLocation.latitude).toFixed(4)}, {Number(selectedLocation.longitude).toFixed(4)}
          </div>
        </div>
      )}

      <MapContainer
        center={center}
        zoom={zoom}
        className="w-full h-full"
        zoomControl={false}
      >
        <TileLayer url={tileLayerUrl} attribution="© Map Data" />

        <MapController
          selectedLocation={selectedLocation}
          center={currentCenter}
          onZoomChange={setCurrentZoom}
          onCenterChange={setCurrentCenter}
        />

        {/* --- 4. Render Roads if data exists --- */}
        {/* We use a key to force re-render when the data changes, usually React-Leaflet needs this */}
        {roadData && (
          <GeoJSON
            key={JSON.stringify(roadData).length} // Simple hash to force update when data changes
            data={roadData}
            style={roadStyle}
          />
        )}

        {locations.map((location) => {
          if (!location.latitude || !location.longitude) return null;

          return (
            <Marker
              key={location.id}
              position={[Number(location.latitude), Number(location.longitude)]}
              icon={getMarkerIcon(
                location.category,
                selectedLocation?.id === location.id
              )}
              eventHandlers={{
                click: () => handleMarkerClick(location),
              }}
            >
              <Popup>
                <div className="p-1">
                  <h3 className="font-bold text-sm">{location.name}</h3>
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
    </div>
  );
}
