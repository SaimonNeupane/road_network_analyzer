import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import React from "react";
import type { LatLngExpression } from "leaflet";

const MapView: React.FC = () => {
  const center: LatLngExpression = [27.6253, 85.5561];
  return (
    <MapContainer
      center={center}
      zoom={13}
      style={{ height: "100vh", width: "100%" }}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution="© OpenStreetMap contributors"
      />

      <Marker position={[27.6194, 85.5388]}>
        <Popup>This is KU 🇳🇵</Popup>
      </Marker>
    </MapContainer>
  );
};

export default MapView;
