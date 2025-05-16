

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

interface DropdownProps {
  onValueChange: (value: string) => void;
}

export function Dropdown({ onValueChange }: DropdownProps) {
  return (
    <Select onValueChange={onValueChange}>
      <SelectTrigger className="w-[180px] text-white">
        <SelectValue placeholder="Select a Exercise" />
      </SelectTrigger>
      <SelectContent>
        <SelectGroup className="text-white bg-gray-900">
          <SelectLabel>Exercise</SelectLabel>
          <SelectItem className="hover:bg-gray-800" value="squat">Squats</SelectItem>
          <SelectItem className="hover:bg-gray-800" value="bicep_curl">Bicep Curls</SelectItem>
          <SelectItem className="hover:bg-gray-800" value="plank">Planks</SelectItem>
          <SelectItem className="hover:bg-gray-800" value="push_ups">Push-ups</SelectItem>
          <SelectItem className="hover:bg-gray-800" value="jumping_jacks">Jumping Jacks</SelectItem>
        </SelectGroup>
      </SelectContent>
    </Select>
  )
}
