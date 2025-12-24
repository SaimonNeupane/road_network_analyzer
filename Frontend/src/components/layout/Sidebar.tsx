import SearchBox from "../control/SearchBox";
import RadioFilter from "../control/RadioFilter";
import LocationDescription from "../info/LocationDescription";

interface Props {
  search: string;
  onSearchChange: (v: string) => void;
  onSearchSubmit: () => void;
  selectedRadio: string;
  onRadioChange: (v: string) => void;
  description: string;
}

const Sidebar = ({
  search,
  onSearchChange,
  onSearchSubmit,
  selectedRadio,
  onRadioChange,
  description,
}: Props) => {
  return (
    <div className="w-1/3 p-4 space-y-4">
      <SearchBox
        value={search}
        onChange={onSearchChange}
        onSubmit={onSearchSubmit}
      />

      <RadioFilter selected={selectedRadio} onChange={onRadioChange} />

      <LocationDescription description={description} />
    </div>
  );
};

export default Sidebar;
