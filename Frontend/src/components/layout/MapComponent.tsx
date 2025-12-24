"use client";

import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";

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

export default function MapComponent({
  locations,
  center,
  zoom,
  onMarkerClick,
}: MapComponentProps) {
  return (
    <div className="w-full h-screen rounded-lg overflow-hidden shadow-lg">
      <MapContainer center={center} zoom={zoom} className="w-full h-full">
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="© OpenStreetMap contributors"
        />

        {locations.map((location) => (
          <Marker
            key={location.id}
            position={[location.latitude, location.longitude]}
            eventHandlers={{
              click: () => onMarkerClick(location),
            }}
          >
            <Popup>
              <h3 className="font-bold">{location.name}</h3>
              <p className="text-sm">{location.description}</p>
              <span className="text-xs text-gray-500">{location.category}</span>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
}
