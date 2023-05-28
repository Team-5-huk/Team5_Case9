import React, { useState } from 'react';
import styles from './CreateObject.module.scss';
import CustomInput from '../../../components/CustomInput/CustomInput';
import CustomButton from '../../../components/CustomButton/CustomButton';
import { apiCreateProject } from '../../../api/adminApi/adminApi';
import linksStore from '../../../store/linksStore';
import { observer } from 'mobx-react-lite';

const CreateObject = observer(({ closeModal }) => {
	const {linkGetProjects} = linksStore;
	const [idObj, setIdObj] = useState('');
	const [name, setName] = useState('');
	const [slug, setSlug] = useState('');
	const [latitude, setLatitude] = useState('');
	const [longitude, setLongitude] = useState('');

	const createNewObject = () => {
		const newProject = {
			samolet_pk: idObj,
			name,
			slug,
			latitude,
        	longitude,
		}
		console.log(newProject);
		// apiCreateProject(linkGetProjects, newProject).then(({data, error}) => {
		// 	console.log(data);
		// 	console.log(error);
		// })

		// closeModal();
	};

	return (
		<div>
			<div className={styles.container}>
				<CustomInput type="text" placeholder="id" value={idObj} onChange={(e) => setIdObj(e)} />

				<CustomInput type="text" placeholder="Название" value={name} onChange={(e) => setName(e)} />

				<CustomInput type="text" placeholder="slug" value={slug} onChange={(e) => setSlug(e)} />

				<CustomInput type="text" placeholder="Долгота" value={latitude} onChange={(e) => setLatitude(e)} />

				<CustomInput type="text" placeholder="Ширина" value={longitude} onChange={(e) => setLongitude(e)} />
			</div>
			<CustomButton name="Создать" handleClick={createNewObject} />
		</div>
	);
});

export default CreateObject;
