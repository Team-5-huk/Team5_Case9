import React, { useState } from 'react';
import styles from './CreateFrame.module.scss';
import CustomInput from '../../../../components/CustomInput/CustomInput';
import CustomButton from '../../../../components/CustomButton/CustomButton';

const CreateFrame = ({ closeModal }) => {
	const [idObj, setIdObj] = useState('');
	const [name, setName] = useState('');
	const [countEntries, setCountEntries] = useState('');
	const [countApartaments, setCountApartaments] = useState('');
	const [countFloors, setCountFloors] = useState('');

	const createNewObject = () => {
		console.log('Создать');
		closeModal();
	};

	return (
		<div>
			<div className={styles.container}>
				<CustomInput 
					type="text" 
					placeholder="id" 
					value={idObj} 
					onChange={(e) => setIdObj(e)} 
				/>

				<CustomInput 
					type="text" 
					placeholder="Название" 
					value={name} 
					onChange={(e) => setName(e)} 
				/>

				<CustomInput
					type="text"
					placeholder="Количество подъездов"
					value={countEntries}
					onChange={(e) => setCountEntries(e)}
				/>

				<CustomInput
					type="text"
					placeholder="Количество квартир"
					value={countApartaments}
					onChange={(e) => setCountApartaments(e)}
				/>

				<CustomInput
					type="text"
					placeholder="Количество этажей"
					value={countFloors}
					onChange={(e) => setCountFloors(e)}
				/>
			</div>
			<CustomButton name="Создать" width={'85px'} handleClick={createNewObject} />
		</div>
	);
};

export default CreateFrame;
