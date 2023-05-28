import React from 'react';
import styles from './CustomButton.module.scss';

const CustomButton = ({ name, handleClick }) => {
	return (
		<button className={styles.btn} onClick={handleClick}>
			{name}
		</button>
	);
};

export default CustomButton;
