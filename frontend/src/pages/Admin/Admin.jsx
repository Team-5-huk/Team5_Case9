import React, { useState, useEffect } from 'react';
import styles from './Admin.module.scss';
import { useNavigate } from 'react-router-dom';
import ViewModal from '../../components/ViewModal/ViewModal';
import CreateObject from './CreateObject/CreateObject';
import AllObjects from '../../components/AllObjects/AllObjects';
import { apiGetProjects } from '../../api/api';
import linksStore from '../../store/linksStore';
import { observer } from 'mobx-react-lite';
import store from '../../store/store';

const Admin = observer(() => {
	const {linkGetProjects} = linksStore;
	const {allObjects, setAllObjects} = store;
	const navigate = useNavigate();
	const [isOpenModal, setOpenModal] = useState(false);

	useEffect(() => {
		getObjects(linkGetProjects);
	}, []);

	const getObjects = (url = '') => {
		apiGetProjects(url).then(({data, error}) => {
			setAllObjects(data.map(({samolet_pk, name, card_image
			}) => {
				return {
					id: samolet_pk,
					name,
					photo: card_image
				}
			}));
			// console.log(data);
			console.log(error);
		})
	};

	const openCreateModal = () => {
		setOpenModal(true);
	};

	const closeCreateModal = () => {
		setOpenModal(false);
	};

	const selectObject = (id) => {
		navigate(`/admin/${id}`);
	};

	return (
		<div className={styles.container}>
			<AllObjects 
				data={allObjects}
				title={'Выберите ЖК для отслеживания прогресса'} 
				openCreateModal={openCreateModal} 
				handleSelectObject={(id) => selectObject(id)} 
			/>

			<ViewModal title="Новый объект" isModal={isOpenModal} closeModal={closeCreateModal}>
				<CreateObject closeModal={closeCreateModal} />
			</ViewModal>
		</div>
	);
});

export default Admin;
