import React from 'react';
import styles from './CustomInput.module.scss';

const CustomInput = ({type, placeholder, value, onChange = () => {}}) => {
	return (
		<input
			className={styles.inputItem}
			type={type}
			placeholder={placeholder}
			value={value}
			onChange={(e) => onChange(e.target.value)}
		/>
	);
};

export default CustomInput;
