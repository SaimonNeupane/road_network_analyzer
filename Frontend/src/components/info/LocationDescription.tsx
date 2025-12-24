interface Props {
  description: string;
}

const LocationDescription = ({ description }: Props) => {
  return (
    <div className="p-4 border rounded mt-4">
      <h3 className="font-semibold mb-2">Location Description</h3>
      <p>{description || "Select a location to see details"}</p>
    </div>
  );
};

export default LocationDescription;
