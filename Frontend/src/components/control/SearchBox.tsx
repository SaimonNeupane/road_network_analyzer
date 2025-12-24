interface Props {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
}

const SearchBox = ({ value, onChange, onSubmit }: Props) => {
  return (
    <div className="mb-4">
      <input
        type="text"
        placeholder="Search location here"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full p-2 border rounded"
      />
      <button
        onClick={onSubmit}
        className="mt-2 w-full bg-blue-600 text-white p-2 rounded"
      >
        Search
      </button>
    </div>
  );
};

export default SearchBox;
