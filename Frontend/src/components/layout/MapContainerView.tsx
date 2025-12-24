import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import type { LocationData } from "../../types/location";

interface Props {
  locations: LocationData[];
}

const MapContainerView = ({ locations }: Props) => {
  return (
    <MapContainer
      center={[27.7172, 85.324]} // Kathmandu default
      zoom={13}
      className="h-full w-full rounded"
    >
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

      {locations.map((loc) => (
        <Marker key={loc.id} position={[loc.latitude, loc.longitude]}>
          <Popup>
            <strong>{loc.name}</strong>
            <br />
            {loc.description}
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
};

export default MapContainerView;
