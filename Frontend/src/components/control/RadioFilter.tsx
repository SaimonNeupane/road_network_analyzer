interface Props {
  selected: string;
  onChange: (value: string) => void;
}

const options = ["Radio 1", "Radio 2", "Radio 3"];

const RadioFilter = ({ selected, onChange }: Props) => {
  return (
    <div className="p-4 border rounded">
      {options.map((opt) => (
        <label key={opt} className="flex items-center mb-3 gap-2">
          <input
            type="radio"
            value={opt}
            checked={selected === opt}
            onChange={() => onChange(opt)}
          />
          {opt}
        </label>
      ))}
    </div>
  );
};

export default RadioFilter;
