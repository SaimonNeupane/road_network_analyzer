import { useState } from "react";
import Sidebar from "../components/layout/Sidebar";
import MapContainerView from "../components/layout/MapContainerView";
import type { LocationData } from "../types/location";
import { fetchLocations } from "../services/api";

const MapPage = () => {
  const [search, setSearch] = useState("");
  const [radio, setRadio] = useState("");
  const [locations, setLocations] = useState<LocationData[]>([]);
  const [description, setDescription] = useState("");

  const handleSearch = async () => {
    const data = await fetchLocations(search, radio);
    setLocations(data);
    if (data.length > 0) setDescription(data[0].description);
  };

  return (
    <div className="h-screen flex bg-gray-100">
      <Sidebar
        search={search}
        onSearchChange={setSearch}
        onSearchSubmit={handleSearch}
        selectedRadio={radio}
        onRadioChange={setRadio}
        description={description}
      />

      <div className="flex-1 p-4">
        <MapContainerView locations={locations} />
      </div>
    </div>
  );
};

export default MapPage;
