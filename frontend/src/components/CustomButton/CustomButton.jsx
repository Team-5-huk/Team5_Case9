import React from 'react';
import styles from './CustomButton.module.scss';

const CustomButton = ({ name, width, handleClick }) => {
	return (
		<button 
			style={{width}}
			className={styles.btn} 
			onClick={handleClick}
		>
			{name}
		</button>
	);
};

export default CustomButton;
