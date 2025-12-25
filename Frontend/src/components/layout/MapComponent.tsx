"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  useMap,
  useMapEvents,
  GeoJSON,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// --- Types ---
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
  existingRoads?: any; // Existing OSM Network
  proposedRoads?: any; // Python Analysis Result
}

// --- Sub-Component: Road Layers (Memoized for Performance) ---
const RoadLayers = React.memo(
  ({
    existingRoads,
    proposedRoads,
    mapStyle,
  }: {
    existingRoads: any;
    proposedRoads: any;
    mapStyle: string;
  }) => {
    // Memoize Existing Road Style (Background)
    const existingStyle = useMemo(() => {
      return {
        color: mapStyle === "streets" || mapStyle === "light" ? "#9CA3AF" : "#4B5563",
        weight: 1.5,
        opacity: 0.4,
      };
    }, [mapStyle]);

    // Memoize Proposed Road Style Logic
    const getProposedStyle = useCallback((feature: any) => {
      const type = feature.properties?.type;

      // 1. "new_cycleway" (The Straight Lines) -> BLUE BLOCK
      if (type === "new_cycleway") {
        return {
          color: "#2563EB", // Blue-600
          weight: 6,        // Thick "Block" style
          opacity: 0.9,
          lineCap: "butt" as const,
        };
      }
      // 2. "new_driveway" (The Curved Lines) -> RED DOTTED
      else if (type === "new_driveway") {
        return {
          color: "#DC2626", // Red-600
          weight: 4,
          opacity: 0.8,
          dashArray: "6, 12", // Dotted effect
          lineCap: "round" as const,
        };
      }
      return { color: "#3388ff", weight: 3 };
    }, []);

    // Memoize Popup Logic
    const onEachProposedFeature = useCallback((feature: any, layer: L.Layer) => {
      if (feature.properties) {
        const { rank, score, type } = feature.properties;
        let title = "Proposal";
        let desc = "Road Upgrade";

        if (type === "new_cycleway") {
          title = "🚲 Proposed Cycle Shortcut";
          desc = "Direct point-to-point connection";
        } else if (type === "new_driveway") {
          title = "🛣️ Proposed Road Expansion";
          desc = "Calculated route upgrade";
        }

        layer.bindPopup(`
            <div class="text-sm">
              <strong class="block text-base mb-1 text-gray-800">${title}</strong>
              <span class="text-xs text-gray-500 block mb-2">${desc}</span>
              <hr class="my-1 border-gray-200"/>
              <div class="flex justify-between mt-2">
                <span class="text-gray-600">Rank:</span> 
                <b>#${rank}</b>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-600">AI Score:</span> 
                <b>${Number(score).toFixed(3)}</b>
              </div>
            </div>
          `);
      }
    }, []);

    return (
      <>
        {/* Layer 1: Existing Roads */}
        {existingRoads && (
          <GeoJSON
            key={`existing-${existingRoads.features?.length || 0}`}
            data={existingRoads}
            style={existingStyle}
          />
        )}

        {/* Layer 2: Proposed Roads */}
        {proposedRoads && (
          <GeoJSON
            key={`proposed-${proposedRoads.features?.length || 0}`}
            data={proposedRoads}
            style={getProposedStyle}
            onEachFeature={onEachProposedFeature}
          />
        )}
      </>
    );
  },
  (prevProps, nextProps) => {
    return (
      prevProps.mapStyle === nextProps.mapStyle &&
      prevProps.existingRoads === nextProps.existingRoads &&
      prevProps.proposedRoads === nextProps.proposedRoads
    );
  }
);
RoadLayers.displayName = "RoadLayers";

// --- Helper: Map Controller ---
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

// --- Helper: Icons ---
function getMarkerIcon(category: string, isSelected: boolean) {
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

// --- Main Component ---
export default function MapComponent({
  locations,
  center,
  zoom,
  onMarkerClick,
  existingRoads,
  proposedRoads,
}: MapComponentProps) {
  const [selectedLocation, setSelectedLocation] = useState<LocationData | null>(null);
  const [currentZoom, setCurrentZoom] = useState(zoom);
  const [currentCenter, setCurrentCenter] = useState(center);
  const [mapStyle, setMapStyle] = useState("streets");
  const [showControls, setShowControls] = useState(true);

  const handleMarkerClick = useCallback(
    (location: LocationData) => {
      setSelectedLocation(location);
      onMarkerClick(location);
    },
    [onMarkerClick]
  );

  const tileLayerUrl = useMemo(() => {
    const styles: Record<string, string> = {
      streets: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      satellite:
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      dark: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      light: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    };
    return styles[mapStyle] || styles.streets;
  }, [mapStyle]);

  const getCategoryColor = (category: string) => {
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

      {/* 1. Map Style Controls (Top Right) */}
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

      {/* 2. LEGEND (Bottom Right) */}
      <div className="absolute bottom-6 right-4 z-[1000] bg-white rounded-lg shadow-xl p-3 border border-gray-100 w-56">
        <h4 className="text-xs font-bold text-gray-500 uppercase mb-2 border-b pb-1">Legend</h4>

        {/* Item 1: Cycle Shortcut */}
        <div className="flex items-center gap-3 mb-2">
          <div className="w-8 h-2 bg-blue-600 rounded-sm"></div>
          <span className="text-xs font-medium text-gray-700">Cycle Shortcut</span>
        </div>

        {/* Item 2: Road Expansion */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-1 border-b-2 border-red-600 border-dashed"></div>
          <span className="text-xs font-medium text-gray-700">Road Expansion</span>
        </div>
      </div>

      {/* 3. Selected Location Popup (Bottom Left) */}
      {selectedLocation && (
        <div className="absolute bottom-4 left-4 z-[1000] w-80 bg-white rounded-lg shadow-xl p-4 animate-in fade-in slide-in-from-bottom-4 duration-300">
          <button
            onClick={() => setSelectedLocation(null)}
            className="absolute top-2 right-2 text-gray-400 hover:text-gray-600"
          >
            ✕
          </button>
          <h3 className="font-bold text-lg mb-2 text-gray-800">
            {selectedLocation.name}
          </h3>
          <p className="text-sm text-gray-600 mb-3 leading-relaxed">
            {selectedLocation.description}
          </p>
          <div className="flex items-center gap-2 mb-3">
            <span
              className={`text-xs text-white px-3 py-1 rounded-full capitalize ${getCategoryColor(
                selectedLocation.category
              )}`}
            >
              {selectedLocation.category}
            </span>
          </div>
        </div>
      )}

      {/* 4. Map Container */}
      <MapContainer
        center={center}
        zoom={zoom}
        className="w-full h-full"
        zoomControl={false}
        preferCanvas={true} // Performance Optimization
      >
        <TileLayer url={tileLayerUrl} attribution="© Map Data" />

        <MapController
          selectedLocation={selectedLocation}
          center={currentCenter}
          onZoomChange={setCurrentZoom}
          onCenterChange={setCurrentCenter}
        />

        <RoadLayers
          existingRoads={existingRoads}
          proposedRoads={proposedRoads}
          mapStyle={mapStyle}
        />

        {locations.map((location) => {
          if (!location.latitude || !location.longitude) return null;
          return (
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
                <div className="p-1 font-bold">{location.name}</div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
    </div>
  );
}
